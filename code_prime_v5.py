import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import networkx as nx
from math import gcd

# Constants
MAX_DISTANCE = 50  # Max distance a vehicle travels (km)
MIN_GROUP_SIZE, MAX_GROUP_SIZE = 1, 50
MIN_TAXI_CAPACITY, MAX_TAXI_CAPACITY = 7, 50
TRIP_DURATION = 15  # Avg trip duration (minutes)
COMPATIBILITY_THRESHOLD = 1
LAMBDA = 0.5  # Penalty weight for unmet capacity

# Generate synthetic data for passengers and taxis
def generate_data(num_passengers, num_taxis):
    return (
        np.random.randint(MIN_GROUP_SIZE, MAX_GROUP_SIZE, num_passengers),
        np.random.randint(MIN_TAXI_CAPACITY, MAX_TAXI_CAPACITY, num_taxis),
        np.random.uniform(0, MAX_DISTANCE, num_passengers),
        np.random.uniform(0, MAX_DISTANCE, num_taxis),
    )

# Compute compatibility and cost matrices
def compute_matrices(group_sizes, capacities, passenger_locs, taxi_locs):
    num_passengers, num_taxis = len(group_sizes), len(capacities)
    compatibility = np.zeros((num_passengers, num_taxis))
    cost = np.zeros((num_passengers, num_taxis))

    for i in range(num_passengers):
        for j in range(num_taxis):
            compatibility[i, j] = gcd(group_sizes[i], capacities[j])
            if compatibility[i, j] >= COMPATIBILITY_THRESHOLD:
                distance = abs(passenger_locs[i] - taxi_locs[j])
                penalty = max(0, group_sizes[i] - capacities[j]) / capacities[j]
                cost[i, j] = (distance / (compatibility[i, j] + 1)) + LAMBDA * penalty
            else:
                cost[i, j] = np.inf
    return compatibility, cost

# Visualize the compatibility graph
def visualize_graph(group_sizes, capacities, compatibility, iteration):
    G = nx.Graph()
    for i, size in enumerate(group_sizes):
        G.add_node(f"P{i+1}", group_size=size, bipartite=0)
    for j, capacity in enumerate(capacities):
        G.add_node(f"T{j+1}", capacity=capacity, bipartite=1)
    for i in range(len(group_sizes)):
        for j in range(len(capacities)):
            if compatibility[i, j] > 0:
                G.add_edge(f"P{i+1}", f"T{j+1}", weight=compatibility[i, j])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True,
            node_color=['skyblue' if n[1]['bipartite'] == 0 else 'lightgreen' for n in G.nodes(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title(f"Compatibility Graph (Iteration {iteration})")
    plt.savefig(f"compatibility_graph_{iteration}.png")
    plt.close()

# Simulate ride allocation
def simulate_ride_allocation(num_passengers, num_taxis, iteration):
    group_sizes, capacities, passenger_locs, taxi_locs = generate_data(num_passengers, num_taxis)
    compatibility, cost = compute_matrices(group_sizes, capacities, passenger_locs, taxi_locs)
    row_ind, col_ind = linear_sum_assignment(cost)
    total_ridership, total_eVMT, total_VMT, total_wait_time = 0, 0, 0, 0

    for i, j in zip(row_ind, col_ind):
        if compatibility[i, j] >= COMPATIBILITY_THRESHOLD:
            total_ridership += group_sizes[i]
            distance = abs(passenger_locs[i] - taxi_locs[j])
            total_VMT += distance
            if group_sizes[i] > capacities[j]:
                total_eVMT += distance
            total_wait_time += (distance / MAX_DISTANCE) * TRIP_DURATION
    avg_wait_time = total_wait_time / num_passengers
    visualize_graph(group_sizes, capacities, compatibility, iteration)
    return total_ridership, total_eVMT, total_VMT, avg_wait_time

# Add a trend line to the plot
def add_trend_line(x, y, ax, degree=1, color='red'):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    ax.plot(x, poly(x), color=color, linestyle='--', label='Trend Line')

# Run experiment and visualize results
def run_experiment():
    num_passengers, num_taxis, iterations = 100, 50, 50
    ridership, eVMT, VMT, wait_times = [], [], [], []

    for iteration in range(1, iterations + 1):
        r, e, v, w = simulate_ride_allocation(num_passengers, num_taxis, iteration)
        ridership.append(r)
        eVMT.append(e)
        VMT.append(v)
        wait_times.append(w)

    x = np.arange(1, iterations + 1)
    metrics = {'Ridership': ridership, 'Empty VMT': eVMT, 'VMT': VMT, 'Wait Times': wait_times}
    colors = ['blue', 'red', 'green', 'orange']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, (key, data) in enumerate(metrics.items()):
        ax = axs[i // 2, i % 2]
        ax.plot(x, data, label=key, color=colors[i])
        add_trend_line(x, data, ax)
        ax.set_title(f'{key} Over Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(key)
        ax.legend()

    plt.tight_layout()
    plt.savefig("experiment_results.png")
    plt.close()

# Execute the experiment
if __name__ == "__main__":
    run_experiment()
