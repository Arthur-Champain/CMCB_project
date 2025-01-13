from modules.GNNmodel_curved import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

import torch
from sklearn.preprocessing import StandardScaler


def train(train_dataset, epoch, model, criterion, optimizer, scaler_y_1, scaler_y_2):
    model.train()
    pred_truth = []
    for data in tqdm(train_dataset, desc=f'Epoch {epoch}', unit=' batch', leave=False):
        try:
            # Forward pass
            out1_scaled, out2_scaled = model(data.batch, data.x, data.edge_index, data.edge_attr)
            out = torch.cat((out1_scaled, out2_scaled), dim=1)
            out = out.squeeze(-1)

            # Get targets
            target_1 = data.y[:, 0]
            target_2 = data.y[:, 1]

            
            # Scale targets - convert to numpy, scale, then back to tensor
            target_1_scaled = torch.from_numpy(
                scaler_y_1.transform(target_1.detach().cpu().numpy().reshape(-1, 1))
            ).float().to(target_1.device)
            
            target_2_scaled = torch.from_numpy(
                scaler_y_2.transform(target_2.detach().cpu().numpy().reshape(-1, 1))
            ).float().to(target_2.device)
            
            
            target = torch.cat((target_1_scaled, target_2_scaled), dim=1)

            # Compute loss and backpropagate
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # For recording predictions and targets, first convert to numpy
            target_1_original = torch.from_numpy(
                scaler_y_1.inverse_transform(target_1_scaled.cpu().numpy())
            ).float()
            
            target_2_original = torch.from_numpy(
                scaler_y_2.inverse_transform(target_2_scaled.cpu().numpy())
            ).float()

            out1_descaled = torch.from_numpy(
                scaler_y_1.inverse_transform(out1_scaled.detach().cpu().numpy())
            ).float()
            
            out2_descaled = torch.from_numpy(
                scaler_y_2.inverse_transform(out2_scaled.detach().cpu().numpy())
            ).float()
            
            # Now concatenate the tensors
            target_original = torch.cat((target_1_original, target_2_original), dim=1)
            out_descaled = torch.cat((out1_descaled, out2_descaled), dim=1)
            
            
            # Save predictions
            pred_truth.append([target_original, out_descaled])

        except Exception as e:
            print(f"\nWarning: {str(e)}. Skipping this batch.")
            continue

    return pred_truth

def test(input_loader, model, criterion, scaler_y_1, scaler_y_2):
    model.eval()
    total_loss = 0
    pred_truth = []
    
    for data in input_loader:
        try:
            # Forward pass
            out1_scaled, out2_scaled = model(data.batch, data.x, data.edge_index, data.edge_attr)
            out = torch.cat((out1_scaled, out2_scaled), dim=1)
            out = out.squeeze(-1)

            # Get targets
            target_1 = data.y[:, 0]
            target_2 = data.y[:, 1]
            
            # Scale targets
            target_1_scaled = torch.from_numpy(
                scaler_y_1.transform(target_1.detach().cpu().numpy().reshape(-1, 1))
            ).float().to(target_1.device)
            
            target_2_scaled = torch.from_numpy(
                scaler_y_2.transform(target_2.detach().cpu().numpy().reshape(-1, 1))
            ).float().to(target_2.device)
            
            
            target = torch.cat((target_1_scaled, target_2_scaled), dim=1)

            # Compute loss
            loss = criterion(out, target)
            total_loss += loss.item()

            
            
            # Convert predictions back to original scale
            target_1_original = torch.from_numpy(
                scaler_y_1.inverse_transform(target_1_scaled.cpu().numpy())
            ).float()
            
            target_2_original = torch.from_numpy(
                scaler_y_2.inverse_transform(target_2_scaled.cpu().numpy())
            ).float()

            out1_descaled = torch.from_numpy(
                scaler_y_1.inverse_transform(out1_scaled.detach().cpu().numpy())
            ).float()
            
            out2_descaled = torch.from_numpy(
                scaler_y_2.inverse_transform(out2_scaled.detach().cpu().numpy())
            ).float()

            # Concatenate tensors
            target_original = torch.cat((target_1_original, target_2_original), dim=1)
            out_descaled = torch.cat((out1_descaled, out2_descaled), dim=1)
            
            pred_truth.append([target_original, out_descaled])

        except Exception as e:
            print(f"\nWarning: {str(e)}. Skipping this batch.")
            continue

    return total_loss / len(input_loader), pred_truth

def run(dataset, train_loader, test_loader, valid_loader, epochs, model_type):
    model = setup(model_type, node_dim=dataset.num_node_features, edge_dim=dataset[0].edge_attr.shape[1], hidden_channels=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Initialize scalers
    scaler_y_1 = StandardScaler()
    scaler_y_2 = StandardScaler()

    # Fit the scalers on the training data
    # Convert train labels into a single numpy array
    train_labels_y_1 = torch.cat([data.y[:, 0] for data in train_loader]).cpu().numpy().reshape(-1, 1)  # 1st target
    train_labels_y_2 = torch.cat([data.y[:, 1] for data in train_loader]).cpu().numpy().reshape(-1, 1)  # 2nd target

    # Fit the scalers
    scaler_y_1.fit(train_labels_y_1)
    scaler_y_2.fit(train_labels_y_2)

    train_losses, test_losses = [], []
    train_pred_truth, test_pred_truth, val_pred_truth = [], [], []
    
    for epoch in range(1, epochs + 1):
        train(train_loader, epoch, model, criterion, optimizer, scaler_y_1, scaler_y_2)
        with torch.no_grad():
            test_loss, _ = test(test_loader, model, criterion, scaler_y_1, scaler_y_2)
            train_loss, _ = test(train_loader, model, criterion, scaler_y_1, scaler_y_2)

        test_losses.append(test_loss)
        train_losses.append(train_loss)
        
        scheduler.step(train_loss)

        last_lr = float(scheduler.get_last_lr()[0])
        

        print(f'\nEpoch: {epoch:03d}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Learning Rate: {last_lr:.7f}')

        if last_lr < 1.5e-5:
            print(f"Training interrupted at epoch {epoch} with an lr of {last_lr}")
            break

    # After training, reverse scaling to get the predictions back to the original scale
    _, val_vals = test(valid_loader, model, criterion, scaler_y_1, scaler_y_2)
    val_pred_truth.extend(val_vals)

    _, train_vals = test(train_loader, model, criterion, scaler_y_1, scaler_y_2)
    train_pred_truth.extend(train_vals)

    _, test_vals = test(test_loader, model, criterion, scaler_y_1, scaler_y_2)
    test_pred_truth.extend(test_vals)

    # Make dictionary with train, test and val_pred_truth
    pred_truth_dic = {'train': train_pred_truth, 'test': test_pred_truth, 'val': val_pred_truth}

    return model, train_losses, test_losses, pred_truth_dic, epoch
