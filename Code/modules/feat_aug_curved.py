#feature augmentation

from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
import networkx as nx
import torch
import numpy as np
from scipy.linalg import eigh
import sys

# Augment node features with degrees
def degree_aug(edge_index, node_features):
    num_nodes = len(node_features)
    node_degrees = degree(edge_index[0], num_nodes=num_nodes)

    aug_features = []
    for n, features in enumerate(node_features):

        # Use clone().detach() to avoid the warning
        degree_tensor = node_degrees[n].view(1).clone().detach()
        augmented_node_features = torch.cat([features, degree_tensor], dim=-1)
        aug_features.append(augmented_node_features)

    return torch.stack(aug_features)

# Augment edge features with edge lengths
def edge_length_aug(edge_index, node_features):
    edge_lengths = torch.norm(node_features[edge_index[0]] - node_features[edge_index[1]], dim=1)

    # Use clone().detach() to avoid the warning
    return edge_lengths.view(-1, 1).clone().detach()

#augments edge features with relative unit vectors
def compute_edge_uvector(x, edge_index, edge_attr):
    
    num_nodes = x.size(0)
    nodes_coordinates = x[:, :2]

    src_nodes = nodes_coordinates[edge_index[0]]
    dst_nodes = nodes_coordinates[edge_index[1]]

    # Compute edge vectors
    edge_vectors = dst_nodes - src_nodes

    # Normalize edge vectors
    edge_lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
    normalized_vectors = edge_vectors / edge_lengths
    uvector_diff = []
    for i in range(num_nodes):
        source_node = i
        edges_with_source_i = torch.where(edge_index[0] == source_node)[0]

        for source_node in edges_with_source_i:

            uvector_diff.append(normalized_vectors[source_node] - normalized_vectors[edges_with_source_i[0]])

    uvector_diff_tensor = torch.stack(uvector_diff)

    concatenated_edge_attr = torch.cat([edge_attr,uvector_diff_tensor], dim=1)

    return concatenated_edge_attr

def compute_degree_centr(node_features, edge_index, y):
    x = node_features[:, :2]
    data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    edge_attr=None
                )
    #create a networkx graph for eigenvector centrality calc (for radius, delaunay, voronoi and knn)
    G = to_networkx(data)
    #compute eigenvector centrality
    degree_centrality = nx.degree_centrality(G)
    
    aug_features = []

    for n, features in enumerate(node_features):
        centrality_tensor = torch.tensor([degree_centrality[n]], dtype=torch.float32) 
        augmented_node_features = torch.cat([features, centrality_tensor], dim=-1)
        aug_features.append(augmented_node_features)

    return torch.stack(aug_features)

def compute_closeness_centrality(node_features, edge_index, edge_attr, y):
    x = node_features[:, :2]

    # Calculate edge lengths (weights) based on Euclidean distance
    src, dst = edge_index
    edge_weights = torch.norm(x[src] - x[dst], dim=1)  # Euclidean distance

    # Prepare PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        edge_attr=edge_attr
    )

    # Convert to NetworkX graph without edge attributes first
    G = to_networkx(data, to_undirected=True)

    # Add weights to edges
    edges = edge_index.t().cpu().numpy()
    for idx, (u, v) in enumerate(edges):
        G.add_edge(u, v, weight=float(edge_weights[idx]))

    # Calculate closeness centrality with weights
    closeness_centrality = nx.closeness_centrality(G, distance='weight')

    # Augment node features with closeness centrality
    aug_features = []
    for n, features in enumerate(node_features):
        centrality_tensor = torch.tensor([closeness_centrality.get(n, 0)], 
                                         dtype=node_features.dtype,
                                         device=node_features.device)
        augmented_node_features = torch.cat([features, centrality_tensor], dim=-1)
        aug_features.append(augmented_node_features)
    
    return torch.stack(aug_features)

#get k closest points around node of interest -> local neighborhood
def _get_local_neighborhood(point_idx, points, k):
    """
    Extract k nearest neighbors for a given point based on Euclidean distance
    
    Args:
        point_idx (int): Index of the central point
        points (torch.Tensor): Point coordinates (N x 2)
        k (int): Number of nearest neighbors to return
    
    Returns:
        np.ndarray: Indices of k nearest neighbors including the central point
    """
    # Calculate distances from point_idx to all other points
    
    center = points[point_idx].unsqueeze(0)
    distances = torch.cdist(center, points).squeeze()
    
    # Get k nearest neighbors (including the point itself)
    k_nearest = torch.topk(distances, k, largest=False)
    
    # Convert to numpy array
    neighborhood = k_nearest.indices.cpu().numpy()
    
    return neighborhood

def estimate_rotation_invariant_curvature(node_features, k):
    points = node_features[:, :2]
    points_np = points.numpy()
    augmented_features = []
    
    for point_idx, features in enumerate(node_features):
        
        local_neighbors = _get_local_neighborhood(point_idx, points, k)        
        
        local_points = points_np[local_neighbors]
        centered_points = local_points - local_points.mean(axis=0)
        
        # Compute covariance matrix
        local_cov = np.cov(centered_points.T)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = eigh(local_cov)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        
        if len(eigenvalues) >= 2:
            # Proper curvature using ratio of eigenvalues
            mean_curvature = eigenvalues[1] / eigenvalues[0]  # Rotation invariant ratio
            mean_curvature_tensor = torch.tensor([mean_curvature], dtype=features.dtype, device=features.device)
            
        else:
            mean_curvature_tensor = torch.tensor([0.0], dtype=features.dtype, device=features.device)
        
        augmented_node_features = torch.cat([features, mean_curvature_tensor], dim=-1)
        
        augmented_features.append(augmented_node_features)
        
    
    return torch.stack(augmented_features)


    