import os 
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import warnings


#dict_keys(['network_points', 'network_core_points', 'network_labels', 'network_fiber_labels', 'point_to_core_mapping', 'core_point_labels', 'fiber_lengths', 'fiber_curvatures', 'theoretical_curvatures', 'analysis', 'params'])

def filter_noise(sim_data):
    points = sim_data['network_points']
    labels = sim_data['network_fiber_labels']

    mask = labels.squeeze() != -1

    filtered_points = points[mask]
    filtered_labels = labels[mask]

    sim_data['network_points'] = filtered_points
    sim_data['network_fiber_labels'] = filtered_labels
    return sim_data

def SPP_dataset_build(sim_dict):
    #extract individual fiber graphs
    #for every i fiber in sim_nb -> graph_nb_i

    
    # Initialize the output datasets for each label
    SPP_dataset = {}

    # Iterate through the simulations in sim_dict
    
    for sim_name, sim_data in tqdm(sim_dict.items(), desc = 'Processing simulation files', unit = 'simulation', leave = False):
        points = sim_data['network_points']
        labels = sim_data['network_fiber_labels'].squeeze()  # Remove extra dimension if present
        lengths = sim_data['fiber_lengths']
        curvatures = sim_data['fiber_curvatures']
        
        # Get unique labels
        unique_labels = torch.unique(labels).tolist()
        
        for n, label in enumerate(unique_labels):
            # Filter points and labels for the current graph
            label_mask = (labels == label)
            fiber_points = points[label_mask]
            
            fiber_length = torch.tensor([lengths[n]], dtype=torch.float32).unsqueeze(0)
            fiber_curvature = torch.tensor([curvatures[n]], dtype=torch.float32).unsqueeze(0)
            
            #concatenate fiber lengths and curvature into 1 torch tensor
            predict_labels = torch.cat((fiber_length, fiber_curvature), dim = 1)  
                  
            
            # Create a PyTorch Geometric Data object for the label -> change the label to fiber length 
            SPP_data = Data(x=fiber_points, y=predict_labels)

            SPP_dataset[f'{sim_name}_{int(label)}'] = SPP_data

            f'{sim_name}_{int(label)}'

    return SPP_dataset

def fiber_sim_import(directory, nb_sims, curvType):
    #directory = r'C:\Users\Arthur Champain\Downloads\Arthur-20241118T122838Z-001\Arthur'


    if curvType == 'clean':
        filelist = [os.path.join(directory, f) for f in os.listdir(directory) if not f.endswith('1.pt')]
    elif curvType == 'noisy':
        filelist = [os.path.join(directory, f) for f in os.listdir(directory) if not f.endswith('0.pt')]
    else: 
        filelist = [os.path.join(directory, f) for f in os.listdir(directory)]

    sim_dict = {}

    for n, file_path in tqdm(enumerate(filelist), desc='Importing simulation files', unit='file'):
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        if n in range(nb_sims):

            if file_extension == '.pt':
                #import and filter noise
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    sim_dict[file_name] = filter_noise(torch.load(file_path))

            elif file_extension == '.csv':
                descriptor = pd.read_csv(file_path, header=0)

    SPP_dataset = SPP_dataset_build(sim_dict)
    
    return SPP_dataset

