import math
import os
import shutil
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import torch


JSON_DIR_PATH = '../json'
GRAPH_OUTPUT_DIR = '../graph_adj_matrix'
THRESHOLD = 50
MAX_DEGREE = 10


def open_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def draw_graph(coordinates, adj_matrix):
    G = nx.Graph()

    for node, coords in coordinates.items():
        G.add_node(int(node), pos=coords)

    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                G.add_edge(i+1, j+1)

    node_positions = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos=node_positions, with_labels=False,
            node_size=1, node_color='lightblue')
    plt.axis('off')
    plt.show()


def get_centroids():
    centroids_per_file = {}
    for json_file_name in os.listdir(JSON_DIR_PATH):
        json_path = os.path.join(JSON_DIR_PATH, json_file_name)
        json_content = open_json(json_path)

        centroid_values = {}
        for key, value in json_content.get("nuc", {}).items():
            centroid = value.get("centroid")
            if centroid is not None:
                centroid_values[key] = centroid

        centroids_per_file[json_file_name] = centroid_values
    return centroids_per_file


def save_adj_matrix(adj_matrix, graph_name):
    # Ensure the output folder exists
    if not os.path.exists(GRAPH_OUTPUT_DIR):
        os.makedirs(GRAPH_OUTPUT_DIR)

    # Save the adjacency matrix as a PyTorch tensor
    output_file = os.path.join(GRAPH_OUTPUT_DIR, graph_name + '.pt')
    adj_tensor = adj_matrix.clone().detach()
    torch.save(adj_tensor, output_file)


def generate_graph(node_coordinates, file_name, kNN=False, draw=True):
    node_coordinates_arr = torch.tensor(
        list(node_coordinates.values()), dtype=torch.float32)
    n = node_coordinates_arr.shape[0]

    # Compute pairwise distances using PyTorch operations
    x_diff = (node_coordinates_arr[:, 0].unsqueeze(
        1) - node_coordinates_arr[:, 0])**2
    y_diff = (node_coordinates_arr[:, 1].unsqueeze(
        1) - node_coordinates_arr[:, 1])**2
    distances = torch.sqrt(x_diff + y_diff)

    # Create adjacency matrix based on the distance threshold
    adj_matrix = (distances < THRESHOLD).int()

    # Consider only K nearest neighbors (or degree of a node)
    if kNN:
        # Ensure the maximum degree of each node is at most 'MAX_DEGREE'
        # Keep only upper triangular part to avoid double counting
        adj_matrix = torch.triu(adj_matrix, diagonal=1)
        for i in range(n):
            # Sort nodes by distance and keep the closest 'MAX_DEGREE' neighbors
            sorted_neighbors = torch.argsort(distances[i])
            for j in range(MAX_DEGREE, n):
                adj_matrix[i][sorted_neighbors[j]] = 0
                adj_matrix[sorted_neighbors[j]][i] = 0

    if draw:
        draw_graph(node_coordinates, adj_matrix)

    save_adj_matrix(adj_matrix, file_name)


if __name__ == '__main__':
    centroids_per_file = get_centroids()
    for file_name, centroids in centroids_per_file.items():
        generate_graph(centroids, file_name.split('.')[0])
