import os
import numpy as np
import torch
from torchvision import datasets, transforms

# Define constants
num_clients = 20
client_prefix = "Client"
data_folder = "data/MNIST"
batch_size = 60000  # Total number of training samples in CIFAR-10 is 50000

# Define CIFAR-10 transformations (to convert to tensors)
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-10 dataset (training data only)
cifar10_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Extract the data and labels
X_train = cifar10_train.data  # shape: (50000, 32, 32, 3)
y_train = np.array(cifar10_train.targets)  # shape: (50000,)

# Shuffle the dataset for IID distribution
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# Split the dataset into `num_clients` parts
client_data_size = X_train.shape[0] // num_clients
client_data = [(X_train[i * client_data_size:(i + 1) * client_data_size],
                y_train[i * client_data_size:(i + 1) * client_data_size])
               for i in range(num_clients)]

# Create the folder structure and save data
for i in range(1, num_clients + 1):
    # Folder structure: Client1/data, Client2/data, ..., Client20/data
    client_folder = f"{client_prefix}{i}"
    data_path = os.path.join(client_folder, data_folder)

    # Create the client and data directories
    os.makedirs(data_path, exist_ok=True)

    # Retrieve the corresponding data for this client
    X_client, y_client = client_data[i - 1]

    # Save the data and labels as .npy files
    np.save(os.path.join(data_path, "X.npy"), X_client)
    np.save(os.path.join(data_path, "y.npy"), y_client)

print("Data successfully distributed across clients.")
