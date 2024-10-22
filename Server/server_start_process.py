import os
import uuid
import websockets
import asyncio
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
from tqdm import tqdm
import importlib
import inspect
from pathlib import Path
import sys
from modelUtil import dequantize_tensor, decompress_tensor
import pickle
from DataLoaders.loaderUtil import getDataloader
from utils import create_message, create_message_results, create_result_dict
from modelUtil import get_criterion
# from DBService import db_service
from Scheduler import Scheduler
import time


class JobServer:

    def __init__(self):
        self.new_latencies = None
        self.num_clients = 0
        self.local_weights = []
        self.local_loss = []
        self.q_diff, self.info = [], []
        self.v, self.i, self.s = [], [], []
        self.bytes = []
        self.comp_len = 0

    def load_dataset(self, folder):

        data_test = np.load('data/' + str(folder) + '/X.npy')
        labels = np.load('data/' + str(folder) + '/y.npy')
        return data_test, labels

    def testing(self, model, preprocessing, bs, criterion):

        dataset, labels = self.load_dataset(preprocessing['folder'])
        test_loss = 0
        correct = 0
        test_loader = DataLoader(getDataloader(dataset, labels, preprocessing), batch_size=bs, shuffle=False)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.eval()
        model.to(device)
        print('device ' + str(device))
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            label = np.squeeze(label)
            loss = criterion(output, label)
            test_loss += loss.item() * data.size(0)
            if preprocessing['dtype'] != 'One D':
                _, pred = torch.max(output, 1)
                correct += pred.eq(label.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        if preprocessing['dtype'] != 'One D':
            test_accuracy = 100. * correct / len(test_loader.dataset)
        else:
            test_accuracy = 0

        return test_loss, test_accuracy

    def is_complete_message(self, received_data):
        """Helper function to check if we have received the complete message.
        This is a placeholder, and you should implement your own logic based on
        your protocol (e.g., using message length headers or delimiters)."""

        try:
            # Try to unpickle the data to check if it's complete
            _ = pickle.loads(received_data)
            return True
        except (pickle.UnpicklingError, EOFError):
            # If unpickling fails, the message is not complete
            return False

    # async def connector(self, client_uri, data, client_index, server_socket):
    #     """connector function for connecting the server to the clients. This function is called asynchronously to
    #     1. send process requests to each client
    #     2. calculate local weights for each client separately"""
    #
    #     async with websockets.connect(client_uri, ping_interval=None, max_size=3000000) as websocket:
    #         finished = False
    #         try:
    #             await websocket.send(data)
    #             start = time.time()
    #             while not finished:
    #                 async for message in websocket:
    #                     try:
    #
    #                         data = pickle.loads(message)
    #                         self.bytes.append(len(message))
    #                         if len(data) == 2:
    #                             self.local_weights.append(copy.deepcopy(data[0]))
    #                             self.local_loss.append(copy.deepcopy(data[1]))
    #
    #                         elif len(data) == 3:
    #                             self.q_diff.append(copy.deepcopy(data[0]))
    #                             self.local_loss.append(copy.deepcopy(data[1]))
    #                             self.info.append(copy.deepcopy(data[2]))
    #                             self.comp_len = len(self.q_diff)
    #
    #                         elif len(data) == 4:
    #                             self.v.append(copy.deepcopy(data[0]))
    #                             self.i.append(copy.deepcopy(data[1]))
    #                             self.s.append(copy.deepcopy(data[2]))
    #                             self.local_loss.append(copy.deepcopy(data[3]))
    #                             self.comp_len = len(self.v)
    #
    #                         finished = True
    #                         self.new_latencies[0, client_index] = time.time() - start
    #                         break
    #
    #                     except Exception as e:
    #
    #                         await server_socket.send(message)
    #
    #
    #         except Exception as e:
    #             pass

    async def connector(self, client_uri, data, client_index, server_socket):
        """Connector function for connecting the server to the clients. This function is called asynchronously to
        1. send process requests to each client
        2. calculate local weights for each client separately"""

        async with websockets.connect(client_uri, ping_interval=None, max_size=3000000) as websocket:
            finished = False
            try:
                await websocket.send(data)  # Sending the request to the client
                start = time.time()
                received_data = bytearray()  # Use a bytearray to accumulate the data
                while not finished:
                    async for message in websocket:
                        try:
                            # Accumulate the received data
                            received_data.extend(message)  # Append the message chunk

                            # Check if we've received the complete data
                            if self.is_complete_message(received_data):
                                print('eof received calculating')
                                data = pickle.loads(received_data)  # Unpickle the complete message
                                self.bytes.append(len(received_data))  # Record the size of the received message

                                # Process the received data
                                if len(data) == 2:
                                    self.local_weights.append(copy.deepcopy(data[0]))
                                    self.local_loss.append(copy.deepcopy(data[1]))
                                elif len(data) == 3:
                                    self.q_diff.append(copy.deepcopy(data[0]))
                                    self.local_loss.append(copy.deepcopy(data[1]))
                                    self.info.append(copy.deepcopy(data[2]))
                                    self.comp_len = len(self.q_diff)
                                elif len(data) == 4:
                                    self.v.append(copy.deepcopy(data[0]))
                                    self.i.append(copy.deepcopy(data[1]))
                                    self.s.append(copy.deepcopy(data[2]))
                                    self.local_loss.append(copy.deepcopy(data[3]))
                                    self.comp_len = len(self.v)

                                finished = True
                                self.new_latencies[
                                    0, client_index] = time.time() - start  # Update the latency for this client
                                break
                            print('EOF not recieved')
                        except Exception as e:
                            print('exception ' + str(e))
                            await server_socket.send(message)  # Send error message to the server

            except Exception as e:
                pass

    async def start_job(self, data, websocket):

        print('Initial model deployment at PS (5TONIC)')
        global model

        job_id = uuid.uuid4().hex
        filename = './ModelData/Model.py'

        # os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'rb') as f:
            data['file'] = f.read()

        path_pyfile = Path(filename)
        sys.path.append(str(path_pyfile.parent))
        mod_path = str(path_pyfile).replace(os.sep, '.').strip('.py')
        imp_path = importlib.import_module(mod_path)

        for name_local in dir(imp_path):

            if inspect.isclass(getattr(imp_path, name_local)):
                modelClass = getattr(imp_path, name_local)
                model = modelClass()

        job_data = data['jobData']
        schemeData = job_data['scheme']
        client_list = job_data['general']['clients']
        print('recieved client list ' + str(client_list))
        T = int(schemeData['comRounds'])
        C = float(schemeData['clientFraction']) if 'clientFraction' in schemeData else 1
        schemeData['clientFraction'] = C
        K = int(len(client_list))
        E = int(schemeData['epoch'])
        eta = float(schemeData['lr'])
        B = int(schemeData['minibatch'])
        B_test = int(schemeData['minibatchtest'])
        preprocessing = job_data['preprocessing']
        compress = job_data['modelParam']['compress']
        scheduler_type = schemeData['scheduler']

        latency_avg = int(schemeData['latency_avg']) if scheduler_type == 'latency' else 1
        # db_service.save_job_data(job_data, job_id)

        criterion = get_criterion(job_data['modelParam']['loss'])

        global_weights = model.state_dict()
        train_loss = []
        test_loss = []
        test_accuracy = []
        round_times = []
        total_bytes = []
        # m = max(int(C * K), 1)
        # print('Broadcasting the initial model and hyperparameters to clients 5TONIC1,5TONIC2 , OULU1 and OULU2')
        # run for number of communication rounds
        scheduler = Scheduler(scheduler_type, K, C, avg_rounds=latency_avg)
        for curr_round in range(1, T + 1):
            start_time = time.time()
            # TODO need to check
            print('current communication round ' + str(curr_round))
            # S_t = np.random.choice(range(K), m, replace=False)
            S_t = scheduler.get_workers(self.new_latencies)
            self.new_latencies = np.ones((1, K), dtype='float')
            client_ports = [clt for clt in client_list]
            clients = [client_ports[i] for i in S_t]
            st_count = 0
            print('sending to clients ' + str(clients))
            tasks = []
            if curr_round > 0:
                print('Broadcasting aggregated model to to clients 5TONIC1,5TONIC2 , OULU1 and OULU2')
            for client in clients:
                print('sending to client ' + str(client))
                client_uri = 'ws://' + str(client['client_ip']) + '/process'
                serialized_data = create_message(B, eta, E, data['file'], job_data['modelParam'],
                                                 preprocessing, global_weights)
                client_index = client_ports.index(client)
                tasks.append(self.connector(client_uri, serialized_data, client_index, websocket))
                st_count += 0
            await asyncio.gather(*tasks)
            print('Received model updates from clients')
            if compress:

                for i in range(self.comp_len):

                    count = 0
                    for server_param in model.parameters():
                        if compress == 'quantize':

                            z_point = float(job_data['modelParam']['z_point'])
                            scale = float(job_data['modelParam']['scale'])
                            server_param.data += dequantize_tensor(self.q_diff[i][count], scale, z_point,
                                                                   self.info[i][count]) / len(self.q_diff)
                        else:

                            server_param.data += decompress_tensor(self.v[i][count], self.i[i][count],
                                                                   self.s[i][count]) / len(self.v)
                        count += 1

                global_weights = model.state_dict()

            else:
                print('Aggregating local models')
                # TODO local weights are not reset check it
                weights_avg = copy.deepcopy(self.local_weights[0])
                for k in weights_avg.keys():
                    for i in range(1, len(self.local_weights)):
                        weights_avg[k] += self.local_weights[i][k]

                    weights_avg[k] = torch.div(weights_avg[k], len(self.local_weights))

                global_weights = weights_avg
            # torch.save(model.state_dict(), "./ModelData/" + str(job_id) + '/model.pt')
            torch.save(model.state_dict(), 'model.pt')
            model.load_state_dict(global_weights)

            loss_avg = sum(self.local_loss) / len(self.local_loss)
            train_loss.append(loss_avg)

            g_loss, g_accuracy = self.testing(model, preprocessing, B_test, criterion)
            print('calculated test accuracy at round ' + str(curr_round) + ' = ' + str(g_accuracy))
            test_loss.append(g_loss)
            test_accuracy.append(g_accuracy)
            elapsed_time = round(time.time() - start_time, 2)
            if len(round_times) > 0:
                tot_time = round_times[-1] + elapsed_time
            else:
                tot_time = elapsed_time

            round_times.append(tot_time)
            if len(total_bytes) > 0:
                tot_bytes = total_bytes[-1] + self.bytes[-1] / 1e6
            else:
                tot_bytes = self.bytes[-1] / 1e6
            total_bytes.append(round(tot_bytes, 2))

            with open('test_accuracy.txt', 'a') as fw:
                fw.write(str(test_accuracy) + '\n')

            with open('test_loss.txt', 'a') as fw:
                fw.write(str(test_loss) + '\n')

            with open('elapsed_time.txt', 'a') as fw:
                fw.write(str(round_times) + '\n')

            with open('bytes.txt', 'a') as fw:
                fw.write(str(total_bytes) + '\n')

            if curr_round == T:
                serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round,
                                                            round_times,
                                                            total_bytes, True)
            else:
                serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round,
                                                            round_times,
                                                            total_bytes, False)
            result_dict = create_result_dict(test_accuracy, train_loss, test_loss, curr_round, elapsed_time)
            # db_service.save_results(result_dict, job_id)
            if curr_round == T:
                print('FL training completed. Final model saved at clients')
            await websocket.send(serialized_results)
