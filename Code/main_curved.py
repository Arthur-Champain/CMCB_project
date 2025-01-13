from modules import *

#grid search
#think about way to do adaptive learning for different conditions 
# -> brute force infinite epoch let schedule reduce at stop under 1e-6 ?

#try model only on curvature

#abs of curvature here 
# try loss function percentage of diff -> normalized over 1
# abs(curv - pred_curv) / curv
#abs(length - pred_length) / length
#percentage of error mean squared

#normalize by tot nb of batches / batch size for train set

# Import data
directory = r'C:\Users\Arthur Champain\Downloads\Arthur-20241118T122838Z-001\Curved_SPPs'
nb_sims = 450
epochs = 150
curvType = 'clean'
#curvType = clean, noisy, both
model_type = 'GAT'
#model_type = GAT, GNN
graph_type = 'Voronoi'
condition_name = 'Voronoi_test'
dataset_status =  'build'   
#dataset_status = load, build
#can be Delaunay (good), Voronoi (good), Radius(mid?), knn (slow) or Complete /!\(big files, inhumanly slow)
batch_size = int(nb_sims/10)
r = 30
k = 15


SPP_dataset = data_import_curved.fiber_sim_import(directory, nb_sims, curvType)

dataset = graph_classes_curved.build(dataset_status, SPP_dataset, directory, graph_type, k, r)
#graph_classes.build(Input SPP dataset, directory, graph_type, k/max k = 1, r = 1)

# Load data
train_loader, valid_loader, test_loader = data_split_curved.load(dataset, batch_size, shuffle=True)

# Run model
model, train_losses, test_losses, pred_truth_dic, final_epochs = model_run_curved.run(dataset, train_loader, test_loader, valid_loader, epochs, model_type)


# Plot results (loss curve, truth vs predictions scatter, truth vs predictions diff)
visu_results_curved.plot_main(train_losses, test_losses, final_epochs, pred_truth_dic, graph_type, condition_name)

print('Done!')
'''
#test rotational invariance
rotation_curved.test(model, SPP_dataset, directory, graph_type, k, r)
'''