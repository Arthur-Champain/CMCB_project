from torch_geometric.loader import DataLoader
from torch.utils.data import random_split


def load(dataset, batch_size=2, shuffle=True):

    # Get the total filtered dataset length
    total_length = len(dataset)

    # Define the proportions for splits
    length_test = total_length // 5  # Test + val dataset size (20% of total)
    length_train = total_length - length_test  # Remaining for training (80% of total)


    length_valid = length_test//2 # valset  = 10% of total	
    length_test_final = length_test - length_valid # testset = 10% of total

    # Split the dataset into training and test datasets
    train_dataset, test_dataset, valid_dataset = random_split(dataset, [length_train, length_test_final, length_valid]) 

    # Further split the training dataset into training and validation


    print(f'\nDataset length:{total_length}')
    print(f'Train length:{length_train}')
    print(f'Valid length:{length_valid}')
    print(f'Test length:{length_test_final}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)  

    return train_loader, valid_loader, test_loader