import os
import shutil

# Define constants
num_clients = 20
client_prefix = "Client"
source_code_folder = "code"  # Folder containing the Python files to distribute


# Function to delete any existing 'code' folder inside client directories
def delete_existing_code_folders():
    for i in range(1, num_clients + 1):
        client_folder = f"{client_prefix}{i}"
        code_path_in_client = os.path.join(client_folder, source_code_folder)

        # If the 'code' folder exists in the client folder, delete it
        if os.path.exists(code_path_in_client):
            shutil.rmtree(code_path_in_client)
            print(f"Deleted existing '{source_code_folder}' folder in {client_folder}")


# Function to copy the contents of the code folder to each client's folder
def distribute_code_to_clients():
    # Get the list of all files and subdirectories in the 'code' folder
    if os.path.exists(source_code_folder):
        code_contents = os.listdir(source_code_folder)
    else:
        print(f"Source folder '{source_code_folder}' not found.")
        return

    # Loop through each client folder
    for i in range(1, num_clients + 1):
        client_folder = f"{client_prefix}{i}"

        # Copy each file and subdirectory from 'code' to the root of the client's folder
        for item in code_contents:
            source_path = os.path.join(source_code_folder, item)
            destination_path = os.path.join(client_folder, item)

            # If it's a directory, copy it recursively
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            else:
                # If it's a file, copy the file
                shutil.copy2(source_path, destination_path)

        print(f"Distributed code contents to {client_folder}")


# First, delete any existing 'code' folders in the client directories
# delete_existing_code_folders()

# Then, distribute the contents of 'code' to each client's root folder
distribute_code_to_clients()

print("Code distribution completed successfully.")


# import os
# import shutil
# import numpy as np
#
# # Define constants
# num_clients = 20
# client_prefix = "Client"
# data_folder_name = "data"  # The original data folder where X.npy and y.npy are stored
#
#
# # Function to distribute the CIFAR-10 data to clients
# def distribute_code_to_clients():
#     # Loop through each client folder
#     for i in range(1, num_clients + 1):
#         client_folder = f"{client_prefix}{i}"
#         client_data_folder = os.path.join(client_folder, data_folder_name)
#
#         # Check if the client data folder exists
#         if os.path.exists(client_data_folder):
#             # Create a new folder named 'cifar10' inside the 'data' folder for this client
#             cifar10_folder = os.path.join(client_data_folder, "cifar10")
#             os.makedirs(cifar10_folder, exist_ok=True)
#
#             # Move X.npy and y.npy from the data folder to the new cifar10 folder
#             for filename in ["X.npy", "y.npy"]:
#                 source_path = os.path.join(client_data_folder, filename)
#                 destination_path = os.path.join(cifar10_folder, filename)
#
#                 if os.path.exists(source_path):
#                     shutil.move(source_path, destination_path)
#                     print(f"Moved {filename} to {cifar10_folder}")
#                 else:
#                     print(f"{filename} not found in {client_data_folder}")
#         else:
#             print(f"Data folder not found for {client_folder}")
#
#
# # Call the function to distribute data
# distribute_code_to_clients()
