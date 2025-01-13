import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns   

def save_plot(filename):
    directory = r'C:\Users\Arthur Champain\Downloads\Arthur-20241118T122838Z-001\run_results\Curved\grid_search'
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist
    plt.savefig(os.path.join(directory, filename), bbox_inches='tight') 
    

def plot_loss(train_losses, test_losses, epoch, graph_type, condition_name):
# Plot the loss across epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epoch+1), test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Across Epochs ({graph_type})')
    plt.legend()
    plt.grid(True)
    save_plot(f'{condition_name}_{graph_type}_loss_plot.png')
    return

def extract_pred_truth(pred_truth, index):
    predictions = []
    targets = []
    for subarray in pred_truth:   
    # unpack nested pred_truth
        targetlist, outlist = subarray  # Unpack the lists

        # Extract the target values based on index
        for target in targetlist:
            targets.append(target[index])  # Access the target value based on index
            

        # Extract the prediction values based on index
        for out in outlist:
            predictions.append(out[index])  # Access the prediction value based on index

    return predictions, targets

def plot_pred_truth_scatt(pred_truth_dic, graph_type, condition_name):
    # Extract predictions and targets for length and curvature
    val_length_preds, val_length_targets = extract_pred_truth(pred_truth_dic['val'],index = 0)
    val_curvature_preds, val_curvature_targets = extract_pred_truth(pred_truth_dic['val'],index = 1) 

    test_length_preds, test_length_targets = extract_pred_truth(pred_truth_dic['test'],0)
    test_curvature_preds, test_curvature_targets = extract_pred_truth(pred_truth_dic['test'],1)

    train_length_preds, train_length_targets = extract_pred_truth(pred_truth_dic['train'],0)
    train_curvature_preds, train_curvature_targets = extract_pred_truth(pred_truth_dic['train'],1)

    # Create sets for plotting
    sets = ['train', 'test', 'val']
    colors = {'train': 'red', 'test': 'green', 'val': 'blue'}
    labels = {'train': 'Train', 'test': 'Test', 'val': 'Validation'}

    # Plot for each dataset
    for current_set in sets:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Predictions vs Ground Truth: {labels[current_set]} Set', fontsize=16)

        # Select data based on current set
        if current_set == 'train':
            length_preds, length_targets = train_length_preds, train_length_targets
            curvature_preds, curvature_targets = train_curvature_preds, train_curvature_targets
        elif current_set == 'test':
            length_preds, length_targets = test_length_preds, test_length_targets
            curvature_preds, curvature_targets = test_curvature_preds, test_curvature_targets
        else:  # val set
            length_preds, length_targets = val_length_preds, val_length_targets
            curvature_preds, curvature_targets = val_curvature_preds, val_curvature_targets

        # Determine ranges for equality lines
        length_min = min(min(length_targets), min(length_preds))
        length_max = max(max(length_targets), max(length_preds))
        curvature_min = min(min(curvature_targets), min(curvature_preds))
        curvature_max = max(max(curvature_targets), max(curvature_preds))

        #linear regression
        length_a, length_b = np.polyfit(length_targets, length_preds, 1)
        ax1.scatter(length_targets, length_preds, 
                    color=colors[current_set], 
                    alpha=0.7, 
                    label=f'{labels[current_set]} Predictions', 
                    s=20)
        
        ax1.plot([length_min, length_max], [length_min, length_max], 
                 'k--', label='Equality Line', linewidth=2)
        ax1.set_xlabel('Length Ground Truth (nm)', fontsize=10)
        ax1.set_ylabel('Length Prediction (nm)', fontsize=10)
        ax1.set_title(f'Length: {labels[current_set]} Set\nPrediction fit: y = {length_a:.2f}x + {length_b:.2f}', fontsize=12)
        ax1.legend()
        ax1.grid(False)

        # Curvature subplot
        curv_a, curv_b = np.polyfit(curvature_targets, curvature_preds, 1)
        ax2.scatter(curvature_targets, curvature_preds, 
                    color=colors[current_set], 
                    alpha=0.7, 
                    label=f'{labels[current_set]} Predictions', 
                    s=20)
        ax2.plot([curvature_min, curvature_max], [curvature_min, curvature_max], 
                 'k--', label='Equality Line', linewidth=2)
        ax2.set_xlabel(r'Curvature Ground Truth (nm$^{-1}$)', fontsize=10)
        ax2.set_ylabel(r'Curvature Predictions (nm$^{-1}$)', fontsize=10)
        ax2.set_title(f'Curvature: {labels[current_set]} Set\nPrediction fit: y = {curv_a:.2f}x + {curv_b:.5f}', fontsize=12)
        ax2.legend()
        ax2.grid(False)
        

        # Adjust layout and save
        plt.tight_layout()
        save_plot(f'{condition_name}_{graph_type}_{current_set}_truth_pred_scatt_plot.png')
        plt.close(fig)

    return

def calculate_overlap(hist_target, hist_pred):
    # Calculate the overlap between the two histograms
    hist_target_norm = hist_target / np.sum(hist_target)
    hist_pred_norm = hist_pred / np.sum(hist_pred)

    overlap = np.sum(np.minimum(hist_target_norm, hist_pred_norm))
    
    return overlap

def plot_pred_truth_diff(pred_truth_dic, graph_type, condition_name):
    # Extract length and curvature predictions and targets
    val_length_preds, val_length_targets = extract_pred_truth(pred_truth_dic['val'],0)
    val_curvature_preds, val_curvature_targets = extract_pred_truth(pred_truth_dic['val'],1)

    mean_length_diff = np.mean(np.abs(np.array(val_length_preds) - np.array(val_length_targets)))
    mean_curv_diff = np.mean(np.abs(np.array(val_curvature_preds) - np.array(val_curvature_targets)))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Truths vs Predictions ({graph_type})', fontsize=16)
    
    # Length subplot
    # Sort indices based on target values
    length_sorted_indices = np.argsort(val_length_targets)
    length_targets_sorted = np.array(val_length_targets)[length_sorted_indices]
    length_preds_sorted = np.array(val_length_preds)[length_sorted_indices]

    # Create x positions with a small offset
    x_length = np.arange(len(length_targets_sorted))
    width = 0.4  # Width of each bar

    # Plot length bars with slight horizontal offset
    ax1.bar(x_length, length_targets_sorted, width, label='Target', alpha=0.5, color='blue')
    ax1.bar(x_length, length_preds_sorted, width, label='Predictions', alpha=0.5, color='red')
    #remove the x axis
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel('Length (nm)', fontsize=12)
    
    ax1.set_title('Length: Truths vs Predictions')
    ax1.legend()

    ax1.text(0.5, 0.9, f'Mean prediction-truth difference: {mean_length_diff:.2f}', transform=ax1.transAxes, 
         fontsize=12, color='black', ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # Curvature subplot
    # Sort indices based on target values
    curvature_sorted_indices = np.argsort(val_curvature_targets)
    curvature_targets_sorted = np.array(val_curvature_targets)[curvature_sorted_indices]
    curvature_preds_sorted = np.array(val_curvature_preds)[curvature_sorted_indices]
    x_curv = np.arange(len(curvature_targets_sorted))
    # Plot curvature bars with slight horizontal offset
    ax2.bar(x_curv, curvature_targets_sorted, width, label='Target', alpha=0.5, color='blue')
    ax2.bar(x_curv, curvature_preds_sorted, width, label='Predictions', alpha=0.5, color='red')
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel(r'Curvature (nm$^{-1}$)', fontsize=12)
    
    ax2.set_title('Curvature: Truths vs Predictions')
    ax2.legend()
    ax2.text(0.5, 0.9, f'Mean prediction-truth difference: {mean_curv_diff:.6f}', transform=ax2.transAxes, 
         fontsize=12, color='black', ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout and save
    plt.tight_layout()
    save_plot(f'{condition_name}_{graph_type}_truth_pred_comp_plot.png')
    plt.close(fig)

    return

def plot_pred_truth_distri(pred_truth_dic, graph_type, condition_name, num_bins=20):
    # Extract length and curvature predictions and targets
    val_length_preds, val_length_targets = extract_pred_truth(pred_truth_dic['val'], 0)
    val_curvature_preds, val_curvature_targets = extract_pred_truth(pred_truth_dic['val'], 1)

    # Calculate the minimum and maximum values across both prediction and target datasets
    length_min = min(np.min(val_length_preds), np.min(val_length_targets))
    length_max = max(np.max(val_length_preds), np.max(val_length_targets))
    curvature_min = min(np.min(val_curvature_preds), np.min(val_curvature_targets))
    curvature_max = max(np.max(val_curvature_preds), np.max(val_curvature_targets))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Truth versus Prediction Distributions', fontsize=16)

    # Calculate histograms for length and curvature using the same bins
    hist_length_target, bins_length_target = np.histogram(val_length_targets, bins=np.linspace(length_min, length_max, num_bins+1))
    hist_length_pred, bins_length_pred = np.histogram(val_length_preds, bins=bins_length_target)

    # Length Distribution Histogram
    ax1.bar(bins_length_target[:-1], hist_length_pred, width=np.diff(bins_length_target)*0.9, 
            alpha=0.5, label='Predictions', color='red', edgecolor='none')
    ax1.bar(bins_length_target[:-1], hist_length_target, width=np.diff(bins_length_target)*0.9,
            alpha=0.5, label='Truth', color='blue', edgecolor='none')
    
    length_overlap = calculate_overlap(hist_length_target, hist_length_pred)
    ax1.set_title(f'Length Distribution - Overlap Coefficient: {length_overlap:.2f}', fontsize=12)
    ax1.set_xlabel('Length (nm)')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Curvature Distribution Histogram
    hist_curvature_target, bins_curvature_target = np.histogram(val_curvature_targets, bins=np.linspace(curvature_min, curvature_max, num_bins+1))
    hist_curvature_pred, bins_curvature_pred = np.histogram(val_curvature_preds, bins=bins_curvature_target)

    ax2.bar(bins_curvature_target[:-1], hist_curvature_pred, width=np.diff(bins_curvature_target)*0.9, 
            alpha=0.5, label='Predictions', color='red', edgecolor='none')
    ax2.bar(bins_curvature_target[:-1], hist_curvature_target, width=np.diff(bins_curvature_target)*0.9,
            alpha=0.5, label='Truth', color='blue', edgecolor='none')

    curvature_overlap = calculate_overlap(hist_curvature_target, hist_curvature_pred)
    ax2.set_title(f'Curvature Distribution - Overlap Coefficient: {curvature_overlap:.2f}', fontsize=12)
    ax2.set_xlabel(r'Curvature (nm$^{-1}$)')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Adjust layout and save
    plt.tight_layout()
    save_plot(f'{condition_name}_{graph_type}_truth_pred_distri_plot.png')
    plt.close(fig)

    return

def plot_main(train_losses, test_losses, final_epochs, pred_truth_dic, graph_type, condition_name):
    plot_loss(train_losses, test_losses, final_epochs, graph_type, condition_name)
    plot_pred_truth_scatt(pred_truth_dic, graph_type, condition_name)
    plot_pred_truth_diff(pred_truth_dic, graph_type, condition_name)
    plot_pred_truth_distri(pred_truth_dic, graph_type, condition_name)
    return
