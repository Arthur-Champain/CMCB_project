import os
import numpy as np
import warnings
import shutil
from tqdm import tqdm
from modules.feat_aug_curved import *
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_cluster import knn_graph
from torch_cluster import radius_graph
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from typing import Optional, Callable, Literal

class GraphDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 SPP_dataset: dict,
                 graph_type=str,
                 k = int, 
                 r = int,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """
        Custom PyG Dataset for various graphs types
        
        Args:
            root: Root directory where the dataset should be saved
            SPP_dataset: Dictionary containing the original SPP data
            graph_type: Type of graph to build
            k: Number of nearest neighbors for knn graph
            transform: Optional transform to be applied on each Data object
            pre_transform: Optional transform to be applied on each Data object before saving
        """
        self.SPP_dataset = SPP_dataset
        self.graph_type = graph_type
        self.k = k
        self.r = r
        self.filtered_indices = []  # Initialize early
        super().__init__(root, transform, pre_transform)
        self.filtered_indices = self._filter_valid_graphs()

        if len(self.filtered_indices) == 0:
            raise ValueError("Empty dataset: No valid graphs were built.")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """List of processed file names."""
        return [f'data_{idx}.pt' for idx in self.filtered_indices]

    def download(self):
        pass

    def _filter_valid_graphs(self):
        """Filter out missing or corrupted graphs during initialization."""
        valid_indices = []
        for idx in range(len(self.SPP_dataset)):
            file_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
            if os.path.exists(file_path):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        data = torch.load(file_path)
                    if data is not None:
                        valid_indices.append(idx)
                except Exception as e:
                    print(f"\nSkipping corrupted file {file_path}: {e}")
        return valid_indices

    def _delaunay_edges(self, points):

        tri = Delaunay(points)

        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edges.add((simplex[i], simplex[(i + 1) % 3]))

        edges = np.array(list(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return to_undirected(edge_index)
    
    def _voronoi_edges(self, points):
        
        edges = []
        vor = Voronoi(points)
        # Use ridge_points directly as they represent pairs of points whose cells are adjacent
        for ridge_points in vor.ridge_points:
            # ridge_points contains indices of input points whose cells are adjacent
            edges.append(ridge_points)
        
        edges = np.array(edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return to_undirected(edge_index)

    def _knn_edges(self, points, k, num_points):
        batch = torch.zeros(num_points, dtype=torch.long)
        edge_index = knn_graph(points, k, batch=batch, loop=False)

        return to_undirected(edge_index)
    
    def _radius_edges(self, points, r, k, num_points):
        batch = torch.zeros(num_points, dtype=torch.long)
        edge_index = radius_graph(points, r, batch=batch, loop=False, max_num_neighbors=k)
        
        return to_undirected(edge_index)
    
    def _complete_edges(self, num_points):
        edges_row, edges_col = torch.tril_indices(num_points, num_points, offset=-1)
        # Combine the row and column indices to form the edge_index
        edge_index = torch.stack([edges_row, edges_col], dim=0)

        return to_undirected(edge_index)

    def _feature_augmentation(self, data, graph_type):
        #x_augmented = data.x.clone()
        #edge_attr = torch.zeros(data.edge_index.shape[1], 1)
        edge_attr = edge_length_aug(data.edge_index, data.x)
        
        edge_attr = compute_edge_uvector(data.x, data.edge_index, edge_attr)
        x_augmented = compute_closeness_centrality(data.x, data.edge_index, edge_attr, data.y)

        if graph_type in ['Delaunay', 'Voronoi', 'Radius']:
            
            x_augmented = degree_aug(data.edge_index, x_augmented)
            
            x_augmented = compute_degree_centr(x_augmented, data.edge_index, data.y)
            

        elif graph_type in ['knn', 'Complete']:
            # No degree augmentation for 'knn' and Complete (same node degree all around)
            pass 
        
        num_neighbors = round(data.num_nodes/15) if data.num_nodes >= 45 else 3
        
        x_augmented = estimate_rotation_invariant_curvature(x_augmented, num_neighbors)
        
        x_augmented = x_augmented[:,2:] #remove coordinates from augmented features
        

        data_aug = Data(
                    x=x_augmented.to(torch.float32),
                    edge_index=data.edge_index,
                    edge_attr=edge_attr.to(torch.float32), #edge_attr
                    y=data.y
                )

        return data_aug

    def process(self):
        idx = 0

        for graph_key in tqdm(self.SPP_dataset, desc=f"Building {self.graph_type} dataset", unit="graph", leave = False):
            # Get data for current graph
            SPP_data = self.SPP_dataset[graph_key]
            try:
                points = SPP_data.x.numpy()
                num_points = len(SPP_data.x)
                

                if self.graph_type == 'Delaunay':
                    # Skip if there are not enough points for Delaunay triangulation
                    if points.shape[0] < 4:
                        print(f"\nWarning: Skipping graph '{graph_key}' due to insufficient points for Delaunay triangulation.")
                        idx += 1
                        continue  # Skip to the next graph

                    edge_index = self._delaunay_edges(points)
                
                elif self.graph_type == 'Voronoi':
                    try:
                        edge_index = self._voronoi_edges(points)
                    
                    except Exception as e:
                        print(f"\nWarning: Skipping graph '{graph_key}' due to error in Voronoi diagram computation.")
                        print(e)
                        idx += 1
                        continue  # Skip to the next graph
                
                elif self.graph_type == 'knn':
                    edge_index = self._knn_edges(SPP_data.x, self.k, num_points)

                elif self.graph_type == 'Radius':
                    edge_index = self._radius_edges(SPP_data.x, self.r, self.k, num_points)

                elif self.graph_type == 'Complete':
                    edge_index = self._complete_edges(num_points)

                if edge_index.numel() == 0:
                    print(f"\nWarning: edge_index is empty for graph '{graph_key}'. Skipping this graph.")
                    print(edge_index)
                    continue  
                
                # Create PyG Data object
                data = Data(
                    x=SPP_data.x,
                    edge_index=edge_index,
                    y=SPP_data.y,
                    #pos=SPP_data.x  # Store original positions
                )

                data_aug = self._feature_augmentation(data, self.graph_type)

                if self.pre_transform is not None:
                    data_aug = self.pre_transform(data_aug)

                torch.save(data_aug, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
            
            except Exception as e:
                print(f"\nSkipping graph '{graph_key}' due to error: {e}")
                continue

    def len(self):
        return len(self.filtered_indices)
    
    def get(self, idx):
        try:
            file_path = os.path.join(self.processed_dir, f'data_{self.filtered_indices[idx]}.pt')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data = torch.load(file_path)
            return data
        except IndexError:
            raise ValueError(f"Index {idx} out of range. Check your processed files.")


    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        if data is None:
            print(f"\nWarning: Corrupted or missing data at index {idx}. Skipping.")
            return None  # Allow None values to be filtered later
        return data


class PreBuiltGraphDataset(Dataset):
    # Load a pre-built dataset, mostly used for testing hyperparameters/conditions
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    def len(self):
        # Return the number of files in the dataset directory
        return len([name for name in os.listdir(self.raw_dir) if name.endswith('.pt')])

    def get(self, idx):
        # Load the individual graph
        file_name = f"data_{idx}.pt"
        file_path = os.path.join(self.raw_dir, file_name)
        data = torch.load(file_path)  # Load the graph data (each file is a graph)
        return data

    @property
    def raw_file_names(self):
        # Specify the raw file names (could be empty if not needed)
        return [f"data_{i}.pt" for i in range(self.len())]

    @property
    def processed_file_names(self):
        # Return the names of processed files if you have any (or leave it empty)
        return []


def build(dataset_status,
          SPP_dataset, 
          directory, 
          graph_type: Literal['Delaunay', 'Voronoi', 'knn', 'Radius', 'Complete'], 
          k=0, 
          r=0):
    """
    Create a PyG dataset from SPP data with a specific graph type
    
    Args:
        dataset_status: Status of the dataset. Possible values are 'load' or 'build'
        SPP_dataset: Dictionary containing the original SPP data
        root_dir: Directory to save the processed dataset
        graph_type: Type of graph to build. Possible values are 'Delaunay', 'Voronoi', 'knn', 'Radius', 'Complete'
        k: Number of nearest neighbors (default: 0)
        r: Radius for neighborhood search (default: 0)
    Returns:
        Dataset object
    """    

    dataset_directory = os.path.join(directory,'Graph_datasets',graph_type)  
    if dataset_status == 'load':
        if os.path.exists(dataset_directory):
            graph_dataset = PreBuiltGraphDataset(root=dataset_directory)

            if len(graph_dataset) == 0:
                raise ValueError("Empty dataset. No graphs were loaded.")
            return graph_dataset
        else:
            raise FileNotFoundError(f"No dataset found at '{dataset_directory}'. Please build the dataset first.")
        
    if dataset_status == 'build':
        if os.path.exists(dataset_directory):
            shutil.rmtree(dataset_directory)
        else:
            os.makedirs(dataset_directory, exist_ok=True)

        graph_dataset = GraphDataset(
            root=dataset_directory,
            SPP_dataset=SPP_dataset,
            graph_type=graph_type,
            k=k, 
            r=r
        )

        if len(graph_dataset) ==0:
            raise ValueError("Empty dataset. No graphs were built.")
        
        if (r == 0 or k == 0) and graph_type in ['knn', 'Radius']:
            warnings.warn("\nWarning: k=0 and r=0 can lead to degenerate graphs. Please verify your input.")



    return graph_dataset
