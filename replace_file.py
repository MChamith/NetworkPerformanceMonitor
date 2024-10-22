import os
import shutil

# Define the path to the new ImageDataset.py file
new_image_dataset_path = './code/client_process.py'

# Define the root directory where the client folders are located
clients_root = '.'

# Loop through all client folders
for client_num in range(1, 21):  # Assuming there are 20 clients
    client_folder = f'Client{client_num}'
    client_dataloader_path = os.path.join(clients_root, client_folder)

    # Check if the DataLoaders directory exists inside the client folder
    if os.path.exists(client_dataloader_path):
        # Define the path to the old ImageDataset.py file in the client's folder
        old_image_dataset_path = os.path.join(client_dataloader_path, 'client_process.py')

        # Replace the old file with the new file
        shutil.copy(new_image_dataset_path, old_image_dataset_path)
        print(f"Replaced {old_image_dataset_path} with new ImageDataset.py")
    else:
        print(f"DataLoaders folder not found for {client_folder}. Skipping.")
