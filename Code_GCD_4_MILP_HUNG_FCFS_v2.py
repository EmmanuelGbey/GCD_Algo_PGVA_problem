import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from pulp import *
from scipy.optimize import linear_sum_assignment
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
class Config:
    MAX_DISTANCE = 8.0
    TRIP_DURATION = 15
    BIG_M = 10000
    MU = 0.1  # eVMT linearization coefficient
    DEFAULT_COMPATIBILITY_THRESHOLD = 2
    DEFAULT_ALPHA = 0.33
    DEFAULT_BETA = 0.33
    DEFAULT_GAMMA = 0.34
    EPSILON = 1e-8  # Small value for numerical stability

# -------------------------------------------------------------------
# DATA CLASSES
# -------------------------------------------------------------------
@dataclass
class OptimizationResult:
    assignments: Dict[int, int]
    total_ridership: int
    total_eVMT: float
    total_VMT: float
    avg_wait_time: float
    runtime: float
    status: str
    objective_value: Optional[float] = None
    
    def __post_init__(self):
        """Ensure numerical values are properly formatted"""
        self.total_eVMT = max(0.0, self.total_eVMT)
        self.total_VMT = max(0.0, self.total_VMT)
        self.avg_wait_time = max(0.0, self.avg_wait_time)
        self.runtime = max(0.0, self.runtime)

# -------------------------------------------------------------------
# PRIME FACTOR COMPATIBILITY
# -------------------------------------------------------------------
def prime_factorization(n: int) -> List[int]:
    """Compute prime factorization of integer n"""
    if n <= 1:
        return []
    
    factors = []
    # Handle factor 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Handle odd factors
    divisor = 3
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 2
    
    # If n is still > 1, it's a prime
    if n > 1:
        factors.append(n)
    
    return factors

def compatibility_score(group_size: int, capacity: int) -> float:
    """Calculate compatibility score between group size and vehicle capacity"""
    if group_size <= 0 or capacity <= 0:
        return 0.0
    
    prime_factors_g = prime_factorization(group_size)
    prime_factors_c = prime_factorization(capacity)
    
    if not prime_factors_g or not prime_factors_c:
        return 0.0
    
    set_g = set(prime_factors_g)
    set_c = set(prime_factors_c)
    shared_primes = set_g & set_c
    
    if not shared_primes:
        return 0.0
    
    # GCD component
    gcd_value = gcd(group_size, capacity)
    
    # Jaccard similarity
    jaccard_sim = len(shared_primes) / len(set_g | set_c)
    
    # Frequency bonus for shared prime factors
    total_factors = max(len(prime_factors_g), len(prime_factors_c))
    if total_factors == 0:
        freq_bonus = 0.0
    else:
        freq_bonus = sum(min(prime_factors_g.count(f), prime_factors_c.count(f)) 
                        for f in shared_primes) / total_factors
    
    return gcd_value * (1 + jaccard_sim + freq_bonus)

# -------------------------------------------------------------------
# INPUT VALIDATION
# -------------------------------------------------------------------
def validate_inputs(group_sizes: np.ndarray, capacities: np.ndarray, 
                   passenger_locs: np.ndarray, taxi_locs: np.ndarray) -> bool:
    """Validate input parameters"""
    try:
        # Check dimensions
        n_passengers = len(group_sizes)
        n_taxis = len(capacities)
        
        if n_passengers == 0 or n_taxis == 0:
            raise ValueError("Empty passenger or taxi arrays")
        
        if passenger_locs.shape != (n_passengers, 2):
            raise ValueError(f"passenger_locs shape mismatch: expected ({n_passengers}, 2), got {passenger_locs.shape}")
        
        if taxi_locs.shape != (n_taxis, 2):
            raise ValueError(f"taxi_locs shape mismatch: expected ({n_taxis}, 2), got {taxi_locs.shape}")
        
        # Check value ranges
        if np.any(group_sizes <= 0) or np.any(capacities <= 0):
            raise ValueError("Group sizes and capacities must be positive")
        
        if np.any(passenger_locs < 0) or np.any(taxi_locs < 0):
            raise ValueError("Locations must be non-negative")
        
        return True
        
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return False

# -------------------------------------------------------------------
# MILP Solver (Enhanced)
# -------------------------------------------------------------------
def solve_pgva_milp(group_sizes: np.ndarray, capacities: np.ndarray, 
                   passenger_locs: np.ndarray, taxi_locs: np.ndarray,
                   alpha: float = Config.DEFAULT_ALPHA, 
                   beta: float = Config.DEFAULT_BETA,
                   gamma: float = Config.DEFAULT_GAMMA, 
                   threshold: float = Config.DEFAULT_COMPATIBILITY_THRESHOLD,
                   time_limit: int = 300) -> OptimizationResult:
    """Solve PGVA using MILP with enhanced error handling"""
    
    if not validate_inputs(group_sizes, capacities, passenger_locs, taxi_locs):
        return OptimizationResult({}, 0, 0.0, 0.0, 0.0, 0.0, "Input Error")
    
    n_passengers, n_taxis = len(group_sizes), len(capacities)
    
    try:
        # Precompute compatibility scores and feasibility
        compatibility_matrix = np.zeros((n_passengers, n_taxis))
        feasible_matrix = np.ones((n_passengers, n_taxis), dtype=bool)
        distance_matrix = np.linalg.norm(
            passenger_locs.reshape(-1, 1, 2) - taxi_locs.reshape(1, -1, 2), axis=2
        )

        for i in range(n_passengers):
            for j in range(n_taxis):
                score = compatibility_score(group_sizes[i], capacities[j])
                compatibility_matrix[i, j] = score
                if score < threshold or group_sizes[i] > capacities[j]:
                    feasible_matrix[i, j] = False

        # Create optimization problem
        problem = LpProblem("PGVA_Rideshare", LpMaximize)
        
        # Decision variables
        x = LpVariable.dicts("assignment", 
                           [(i, j) for i in range(n_passengers) for j in range(n_taxis)], 
                           0, 1, cat="Binary")
        y = LpVariable.dicts("taxi_used", range(n_taxis), 0, 1, cat="Binary")
        empty_vmt = LpVariable.dicts("empty_VMT", range(n_taxis), lowBound=0)

        # Objective function
        objective = (
            lpSum(compatibility_matrix[i, j] * group_sizes[i] * x[i, j] 
                  for i in range(n_passengers) for j in range(n_taxis))
            - alpha * lpSum(empty_vmt[j] for j in range(n_taxis))
            - beta * lpSum(distance_matrix[i, j] * x[i, j] 
                          for i in range(n_passengers) for j in range(n_taxis))
            - gamma * lpSum((distance_matrix[i, j] / Config.MAX_DISTANCE) * Config.TRIP_DURATION * x[i, j]
                           for i in range(n_passengers) for j in range(n_taxis))
        )
        problem += objective

        # Constraints
        # Each passenger assigned to at most one taxi
        for i in range(n_passengers):
            problem += lpSum(x[i, j] for j in range(n_taxis)) <= 1

        # Capacity constraints
        for j in range(n_taxis):
            problem += lpSum(group_sizes[i] * x[i, j] for i in range(n_passengers)) <= capacities[j] * y[j]

        # Taxi usage constraints
        for i in range(n_passengers):
            for j in range(n_taxis):
                problem += x[i, j] <= y[j]
                if not feasible_matrix[i, j]:
                    problem += x[i, j] == 0

        # Empty VMT constraints
        for j in range(n_taxis):
            occupied_capacity = lpSum(group_sizes[i] * x[i, j] for i in range(n_passengers))
            problem += empty_vmt[j] >= Config.MU * (capacities[j] * y[j] - occupied_capacity)
            problem += empty_vmt[j] <= Config.BIG_M * y[j]

        # Solve
        start_time = time.time()
        solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        problem.solve(solver)
        runtime = time.time() - start_time

        # Extract results
        assignments = {}
        total_ridership = 0
        total_evmt = 0.0
        total_vmt = 0.0
        total_wait_time = 0.0
        
        status = LpStatus[problem.status]
        objective_value = None
        
        if status == "Optimal":
            objective_value = value(problem.objective)
            
            for i in range(n_passengers):
                for j in range(n_taxis):
                    if x[i, j].varValue and x[i, j].varValue > 0.9:
                        assignments[i] = j
                        total_ridership += group_sizes[i]
                        distance = distance_matrix[i, j]
                        total_vmt += distance
                        total_wait_time += (distance / Config.MAX_DISTANCE) * Config.TRIP_DURATION

            total_evmt = sum(empty_vmt[j].varValue or 0.0 for j in range(n_taxis))
        
        avg_wait_time = total_wait_time / max(1, total_ridership)
        
        return OptimizationResult(
            assignments=assignments,
            total_ridership=int(total_ridership),
            total_eVMT=total_evmt,
            total_VMT=total_vmt,
            avg_wait_time=avg_wait_time,
            runtime=runtime,
            status=status,
            objective_value=objective_value
        )
        
    except Exception as e:
        logger.error(f"MILP solver error: {e}")
        return OptimizationResult({}, 0, 0.0, 0.0, 0.0, 0.0, "Solver Error")

# -------------------------------------------------------------------
# FCFS Solver (Enhanced)
# -------------------------------------------------------------------
def solve_fcfs(group_sizes: np.ndarray, capacities: np.ndarray, 
               passenger_locs: np.ndarray, taxi_locs: np.ndarray) -> OptimizationResult:
    """First-Come-First-Served assignment with improved efficiency"""
    
    if not validate_inputs(group_sizes, capacities, passenger_locs, taxi_locs):
        return OptimizationResult({}, 0, 0.0, 0.0, 0.0, 0.0, "Input Error")
    
    n_passengers, n_taxis = len(group_sizes), len(capacities)
    current_loads = np.zeros(n_taxis)
    assignments = {}
    total_ridership = 0
    total_vmt = 0.0
    total_wait_time = 0.0
    
    start_time = time.time()
    
    try:
        for i in range(n_passengers):
            # Calculate distances to all taxis
            distances = np.linalg.norm(taxi_locs - passenger_locs[i], axis=1)
            
            # Try taxis in order of increasing distance
            for j in np.argsort(distances):
                if current_loads[j] + group_sizes[i] <= capacities[j]:
                    # Assign passenger group to this taxi
                    assignments[i] = j
                    current_loads[j] += group_sizes[i]
                    total_ridership += group_sizes[i]
                    
                    distance = distances[j]
                    total_vmt += distance
                    total_wait_time += (distance / Config.MAX_DISTANCE) * Config.TRIP_DURATION
                    break
        
        # Calculate empty VMT
        total_evmt = sum((capacities[j] - current_loads[j]) * Config.MU for j in range(n_taxis))
        
        runtime = time.time() - start_time
        avg_wait_time = total_wait_time / max(1, total_ridership)
        
        return OptimizationResult(
            assignments=assignments,
            total_ridership=int(total_ridership),
            total_eVMT=total_evmt,
            total_VMT=total_vmt,
            avg_wait_time=avg_wait_time,
            runtime=runtime,
            status="FCFS Complete"
        )
        
    except Exception as e:
        logger.error(f"FCFS solver error: {e}")
        return OptimizationResult({}, 0, 0.0, 0.0, 0.0, 0.0, "FCFS Error")

# -------------------------------------------------------------------
# Hungarian Solver (Enhanced)
# -------------------------------------------------------------------
def solve_hungarian(group_sizes: np.ndarray, capacities: np.ndarray, 
                   passenger_locs: np.ndarray, taxi_locs: np.ndarray,
                   threshold: float = Config.DEFAULT_COMPATIBILITY_THRESHOLD) -> OptimizationResult:
    """Hungarian algorithm assignment with capacity validation"""
    
    if not validate_inputs(group_sizes, capacities, passenger_locs, taxi_locs):
        return OptimizationResult({}, 0, 0.0, 0.0, 0.0, 0.0, "Input Error")
    
    n_passengers, n_taxis = len(group_sizes), len(capacities)
    
    try:
        # Compute compatibility and distance matrices
        compatibility_matrix = np.zeros((n_passengers, n_taxis))
        feasible_matrix = np.ones((n_passengers, n_taxis), dtype=bool)
        distance_matrix = np.linalg.norm(
            passenger_locs.reshape(-1, 1, 2) - taxi_locs.reshape(1, -1, 2), axis=2
        )

        for i in range(n_passengers):
            for j in range(n_taxis):
                score = compatibility_score(group_sizes[i], capacities[j])
                compatibility_matrix[i, j] = score
                if score < threshold or group_sizes[i] > capacities[j]:
                    feasible_matrix[i, j] = False
                    compatibility_matrix[i, j] = -1e9  # Large negative value

        # Create cost matrix (negative compatibility for maximization)
        cost_matrix = -compatibility_matrix
        
        start_time = time.time()
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Validate and extract feasible assignments
        assignments = {}
        total_ridership = 0
        total_vmt = 0.0
        total_wait_time = 0.0
        
        for i, j in zip(row_indices, col_indices):
            if feasible_matrix[i, j]:
                assignments[i] = j
                total_ridership += group_sizes[i]
                distance = distance_matrix[i, j]
                total_vmt += distance
                total_wait_time += (distance / Config.MAX_DISTANCE) * Config.TRIP_DURATION

        # Calculate empty VMT
        current_loads = np.zeros(n_taxis)
        for i, j in assignments.items():
            current_loads[j] += group_sizes[i]
        
        total_evmt = sum((capacities[j] - current_loads[j]) * Config.MU for j in range(n_taxis))
        
        runtime = time.time() - start_time
        avg_wait_time = total_wait_time / max(1, total_ridership)
        
        return OptimizationResult(
            assignments=assignments,
            total_ridership=int(total_ridership),
            total_eVMT=total_evmt,
            total_VMT=total_vmt,
            avg_wait_time=avg_wait_time,
            runtime=runtime,
            status="Hungarian Complete"
        )
        
    except Exception as e:
        logger.error(f"Hungarian solver error: {e}")
        return OptimizationResult({}, 0, 0.0, 0.0, 0.0, 0.0, "Hungarian Error")

# -------------------------------------------------------------------
# EXPERIMENT RUNNER (Enhanced)
# -------------------------------------------------------------------
def run_experiment(num_passengers: int = 50, num_taxis: int = 25, 
                  iterations: int = 30, realistic: bool = True,
                  random_seed: Optional[int] = None) -> Dict:
    """Run comparative experiment with enhanced data collection"""
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    results = {
        'GCD_MILP': {'Ridership': [], 'eVMT': [], 'VMT': [], 'Wait times': [], 'Runtime': [], 'Objective': []},
        'FCFS': {'Ridership': [], 'eVMT': [], 'VMT': [], 'Wait times': [], 'Runtime': [], 'Objective': []},
        'Hungarian': {'Ridership': [], 'eVMT': [], 'VMT': [], 'Wait times': [], 'Runtime': [], 'Objective': []}
    }

    successful_iterations = 0
    
    for iteration in range(1, iterations + 1):
        logger.info(f"Running iteration {iteration}/{iterations}")
        
        # Set seed for reproducibility within iteration
        np.random.seed(iteration if random_seed is None else random_seed + iteration)

        try:
            # Generate data
            if realistic:
                group_sizes = np.random.choice([1, 2, 3, 4, 5], num_passengers, 
                                             p=[0.65, 0.25, 0.08, 0.015, 0.005])
                vehicle_capacities = np.random.choice([ 4, 6], num_taxis, 
                                                    p=[0.8, 0.2])
                passenger_locations = np.random.beta(2, 2, (num_passengers, 2)) * Config.MAX_DISTANCE
                taxi_locations = np.random.beta(2, 2, (num_taxis, 2)) * Config.MAX_DISTANCE
            else:
                group_sizes = np.random.randint(1, 6, num_passengers)
                vehicle_capacities = np.random.randint(1, 6, num_taxis)
                passenger_locations = np.random.uniform(0, Config.MAX_DISTANCE, (num_passengers, 2))
                taxi_locations = np.random.uniform(0, Config.MAX_DISTANCE, (num_taxis, 2))

            # Run all algorithms
            result_milp = solve_pgva_milp(group_sizes, vehicle_capacities, 
                                        passenger_locations, taxi_locations)
            result_fcfs = solve_fcfs(group_sizes, vehicle_capacities, 
                                   passenger_locations, taxi_locations)
            result_hungarian = solve_hungarian(group_sizes, vehicle_capacities, 
                                             passenger_locations, taxi_locations)

            # Store results
            for method, result in [('GCD_MILP', result_milp), ('FCFS', result_fcfs), ('Hungarian', result_hungarian)]:
                results[method]['Ridership'].append(result.total_ridership)
                results[method]['eVMT'].append(result.total_eVMT)
                results[method]['VMT'].append(result.total_VMT)
                results[method]['Wait times'].append(result.avg_wait_time)
                results[method]['Runtime'].append(result.runtime)
                results[method]['Objective'].append(result.objective_value or 0.0)

            successful_iterations += 1
            
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            continue

    logger.info(f"Completed {successful_iterations}/{iterations} iterations successfully")
    return results

# -------------------------------------------------------------------
# ENHANCED VISUALIZER
# -------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def plot_results(metrics: Dict, filename: str = "comparative_results.png"):
        """Create comprehensive results visualization"""
        methods = list(metrics.keys())
        metrics_list = ['Ridership', 'eVMT', 'VMT', 'Wait times', 'Runtime']
        colors = {'GCD MILP': 'blue', 'FCFS': 'red', 'Hungarian': 'green'}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_list):
            ax = axes[idx]
            for method in methods:
                if metric in metrics[method] and metrics[method][metric]:
                    y_values = metrics[method][metric]
                    x_values = np.arange(1, len(y_values) + 1)
                    ax.plot(x_values, y_values, 'o-', label=method, 
                           color=colors.get(method, 'black'), alpha=0.7, markersize=4)
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
        
        # Remove the extra subplot
        fig.delaxes(axes[5])
        
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Results plot saved as {filename}")

    @staticmethod
    def plot_boxplots(metrics: Dict, filename: str = "performance_comparison.png"):
        """Create box plot comparison"""
        methods = list(metrics.keys())
        metrics_list = ['Ridership', 'eVMT', 'VMT', 'Wait times']
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
        
        for idx, metric in enumerate(metrics_list):
            ax = axes_flat[idx]
            values = [metrics[method][metric] for method in methods if metrics[method][metric]]
            
            if values:  # Only plot if we have data
                box_plot = ax.boxplot(values, labels=methods, patch_artist=True)
                for patch, color in zip(box_plot['boxes'], colors[:len(values)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Box plot comparison saved as {filename}")

# -------------------------------------------------------------------
# ENHANCED ANALYZER
# -------------------------------------------------------------------
class Analyzer:
    @staticmethod
    def perform_statistical_analysis(results: Dict):
        """Perform comprehensive statistical analysis"""
        methods = list(results.keys())
        metrics = ['Ridership', 'eVMT', 'VMT', 'Wait times']
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        for metric in metrics:
            print(f"\n{metric.upper().replace('_', ' ')} ANALYSIS:")
            print("-" * 40)
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method_a, method_b = methods[i], methods[j]
                    
                    data_a = results[method_a][metric]
                    data_b = results[method_b][metric]
                    
                    if not data_a or not data_b:
                        continue
                    
                    # Perform t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(data_a, data_b)
                        
                        # Calculate Cohen's d
                        pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a, ddof=1) + 
                                            (len(data_b) - 1) * np.var(data_b, ddof=1)) / 
                                           (len(data_a) + len(data_b) - 2))
                        
                        if pooled_std > Config.EPSILON:
                            cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std
                        else:
                            cohens_d = 0.0
                        
                        # Interpret effect size
                        if abs(cohens_d) < 0.2:
                            effect_size = "small"
                        elif abs(cohens_d) < 0.8:
                            effect_size = "medium"
                        else:
                            effect_size = "large"
                        
                        print(f"{method_a:>10} vs {method_b:<10}: t={t_stat:6.3f}, "
                              f"p={p_value:6.4f}, Cohen's d={cohens_d:6.3f} ({effect_size})")
                              
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {method_a} vs {method_b}: {e}")

    @staticmethod
    def generate_summary(results: Dict):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        metrics = ['Ridership', 'eVMT', 'VMT', 'Wait times', 'Runtime']
        
        for metric in metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 30)
            
            for method in results.keys():
                data = results[method][metric]
                if data:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    median_val = np.median(data)
                    min_val = np.min(data)
                    max_val = np.max(data)
                    
                    print(f"{method:>12}: Mean={mean_val:8.2f}, Std={std_val:7.2f}, "
                          f"Median={median_val:8.2f}, Range=[{min_val:6.2f}, {max_val:6.2f}]")

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
def main():
    """Main execution function"""
    logger.info("Starting Rideshare Optimization Experiment")
    
    # Run experiment with enhanced parameters
    results = run_experiment(
        num_passengers=300, 
        num_taxis=150, 
        iterations=30, 
        realistic=True,
        random_seed=42  # For reproducibility
    )
    
    # Generate visualizations
    Visualizer.plot_results(results, "comparative_results.png")
    Visualizer.plot_boxplots(results, "performance_comparison.png")
    
    # Perform analysis
    Analyzer.perform_statistical_analysis(results)
    Analyzer.generate_summary(results)
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()