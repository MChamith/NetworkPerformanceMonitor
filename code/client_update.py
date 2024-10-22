import json

import torch
from torch.utils.data import DataLoader
import asyncio
from modelUtil import get_optimizer, get_criterion
from DataLoaders.loaderUtil import getDataloader
import numpy as np


class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, labels, optimizer_type, criterion, dataops):
        self.train_loader = DataLoader(getDataloader(dataset, labels, dataops), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.criterion = criterion

    async def train(self, model, websocket):

        criterion = get_criterion(self.criterion)
        optimizer = get_optimizer(self.optimizer_type, model, self.learning_rate)

        e_loss = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device ' + str(device))
        for epoch in range(1, self.epochs + 1):
            print('Current epoch ' + str(epoch))

            train_loss = 0
            model.train()
            for data, labels in self.train_loader:
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                labels = np.squeeze(labels)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)
            local_add = websocket.local_address
            local_add = str(local_add[0]) + ':' + str(local_add[1])
            message = json.dumps(
                {'status': 'training', 'client': str(local_add), 'client_id': 'OULU1', 'epoch': str(epoch)})
            await websocket.send(message)
            await asyncio.sleep(0)
        total_loss = sum(e_loss) / len(e_loss)
        return model.state_dict(), total_loss
