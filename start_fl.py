import asyncio
import os

import websockets
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


async def start_fl():
    algo = 1

    uri = "ws://localhost:8200/job_receive"

    async with websockets.connect(uri) as websocket:
        task_name = 'test task'
        # algo = input("Please select algorithm,\n1.Classic Federated Learning\n2.V-FL Games\n:")
        minibatch = 8
        lr = 0.0001
        epochs = 2

        client_fraction = 1
        minibatch_test = 32
        comm_rounds = 100

        optimizer_no = 1

        if int(optimizer_no) == 1:
            optimizer = 'Adam'
        elif int(optimizer_no) == 2:
            optimizer = 'SGD'
        elif int(optimizer_no) == 3:
            optimizer = 'RMSProp'
        elif int(optimizer_no) == 4:
            optimizer = 'AdaGrad'

        loss_no = 1

        if int(loss_no) == 1:
            loss = 'CrossEntropyLoss'
        elif int(loss_no) == 2:
            loss = 'BCELoss'
        elif int(loss_no) == 3:
            loss = 'MSELoss'

        folder = 'MNIST'
        clients  = []
        for i in range(5):
            ip  = "localhost:" + str(5001 + i)
            client_id = "client" + str(i+1)
            clients.append({"client_ip": str(ip), "client_id": str(client_id)})
        print('clients ' + str(clients))
        if int(algo) == 1:
            job_data = {"jobData": {
                "general": {"task": str(task_name), "method": str(algo), "algo": "Classification",
                            "host": "localhost:8500",
                            "clients": clients,
                            "plots": [None]},
                "scheme": {"minibatch": str(minibatch), "epoch": str(epochs),
                           "lr": str(lr), "scheduler": "random", "clientFraction": str(client_fraction),
                           "minibatchtest": str(minibatch_test),
                           "comRounds": str(comm_rounds)},
                "modelParam": {"optimizer": str(optimizer), "loss": str(loss), "compress": False},
                "preprocessing": {"dtype": "img", "folder": str(folder), "testfolder": str(folder), "normalize": False}}}
        else:
            job_data = {"jobData": {
                "general": {"task": str(task_name), "method": str(algo), "algo": "Classification",
                            "host": "10.5.15.51:8200",
                            "clients": [{"client_ip": "10.5.98.110:5000", "client_id": "5TONIC1"},
                                        {"client_ip": "10.5.98.111:5000", "client_id": "5TONIC2"},
                                        {"client_ip": "87.100.232.38:5000", "client_id": "OULU1"}],
                            "plots": [None]},
                "scheme": {"batch_size": str(minibatch), "epoch": str(epochs),
                           "rep_lr": str(rep_lr),"pred_lr": str(pred_lr), "scheduler": "random", "clientFraction": str(client_fraction),
                           "minibatchtest": str(minibatch_test),
                           "comRounds": str(comm_rounds)},
                "modelParam": {"optimizer": str(optimizer), "loss": str(loss), "compress": False},
                "preprocessing": {"dtype": "img", "folder": str(folder), "testfolder": str(folder),
                                  "normalize": False}}}

        job_data = json.dumps(job_data)
        await websocket.send(job_data)

        test_accuracy = []
        train_loss = []
        test_loss = []
        round_time = []
        total_bytes = []
        final_round = False
        while not final_round:
            async for message in websocket:

                # print(f"<<< {message}")
                message = json.loads(message)

                if message['status'] == 'training':
                    print('Training epoch ' + str(message['epoch']) + ' completed at ' + str(message['client_id']))
                elif message['status'] == 'results':
                    if not message['final']:
                        print('Communication round ' + str(message['round']) + ' completed with accuracy ' + str(
                            message['accuracy']))
                    else:
                        print('Final communication round completed with accuracy ' + str(message['accuracy']))

                if message['status'] == 'results':
                    test_accuracy.append(float(message['accuracy']))
                    train_loss.append(float(message['train_loss']))
                    test_loss.append(float(message['test_loss']))
                    if len(round_time) == 0:
                        round_time.append(float(message['round_time']))
                        total_bytes.append(float(message['total_bytes']))
                    else:
                        r_time = round_time[-1] + float(message['round_time'])
                        t_bytes = total_bytes[-1] + float(message['total_bytes'])
                        round_time.append(r_time)
                        total_bytes.append(t_bytes)

                    final_round = message['final']

                if final_round:
                    rounds = np.array([i for i in range(1, len(test_accuracy) + 1)])

                    font = {
                        'weight': 'bold',
                        'size': 20}

                    matplotlib.rc('font', **font)
                    # print('Current test accuracy ' + str(test_accuracy))
                    fig, axs = plt.subplots(2, figsize=(30, 20))
                    axs[0].plot(rounds, np.array(test_accuracy), label='test accuracy')
                    axs[0].set_title('Number of rounds vs Test Accuracy')
                    axs[0].set(xlabel='Number of Rounds', ylabel='Test Accuracy')
                    axs[1].plot(rounds, np.array(train_loss), label='train loss')
                    axs[1].set_title('Number of rounds vs Train Loss')
                    axs[1].set(xlabel='Number of Rounds', ylabel='Train Loss')

                    # axs[1, 0].plot(rounds, round_time, label='elapsed time')
                    # axs[1, 0].set_title('Number of rounds vs Elapsed time')
                    # axs[1, 0].set(xlabel='Number of Rounds', ylabel='Elapsed time(s)')
                    #
                    # axs[1, 1].plot(rounds, total_bytes, label='total bytes')
                    # axs[1, 1].set_title('Number of rounds vs bytes transferred')
                    # axs[1, 1].set(xlabel='Number of Rounds', ylabel='Total bytes')

                    job_data = json.loads(job_data)
                    task_n = job_data['jobData']['general']['task']
                    os.makedirs(os.path.dirname('results/' + str(task_n)), exist_ok=True)
                    plt.savefig('results/results.png')

                    if int(algo) == 1:
                        np.save('results/classic_test_acc.npy', np.array(test_accuracy))
                        np.save('results/classic_train_loss.npy', np.array(train_loss))

                    if int(algo) == 2:
                        rounds = np.array([i for i in range(1, len(test_accuracy) + 1)])

                        np.save('results/vfl_test_acc.npy', np.array(test_accuracy))
                        np.save('results/vfl_train_loss.npy', np.array(train_loss))

                        classic_acc = np.load('results/classic_test_acc.npy')
                        classic_loss = np.load('results/classic_train_loss.npy')
                        font = {
                            'weight': 'bold',
                            'size': 20}

                        matplotlib.rc('font', **font)
                        # print('Current test accuracy ' + str(test_accuracy))
                        fig, axs = plt.subplots(2, figsize=(30, 20))
                        axs[0].plot(rounds, np.array(test_accuracy), label='V-FL')
                        axs[0].plot(rounds, np.array(classic_acc), label='Classic FL')
                        axs[0].set_title('Number of rounds vs Test Accuracy')
                        axs[0].set(xlabel='Number of Rounds', ylabel='Test Accuracy')
                        axs[1].plot(rounds, np.array(train_loss), label='V-FL')
                        axs[1].plot(rounds, np.array(classic_loss), label='Classic FL')
                        axs[1].set_title('Number of rounds vs Train Loss')
                        axs[1].set(xlabel='Number of Rounds', ylabel='Train Loss')
                        plt.legend(loc="lower right")
                        plt.savefig('results/results_combined.png')


if __name__ == "__main__":
    asyncio.run(start_fl())