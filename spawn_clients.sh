#!/bin/bash

# Number of clients
num_clients=20
client_prefix="Client"
base_port=5000

# Loop through each client folder and spawn a terminal running the Python script
for ((i=1; i<=num_clients; i++))
do
    client_folder="${client_prefix}${i}"
    port_number=$((base_port + i - 1))

    # Check if client_service.py exists in the current client's folder
    if [ -f "${client_folder}/client_service.py" ]; then
        # Open a new xterm and run the Python script with the appropriate port number
        xterm -hold -e "cd ${client_folder}; python client_service.py ${port_number}" &
        echo "Spawned xterm for ${client_folder} with port ${port_number}"
    else
        echo "client_service.py not found in ${client_folder}"
    fi
done
