import copy

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from lib.io.flprint import log_print


class ShareModel:
    def __init__(self, model, criterion, client_epoch, device, fl_print):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.train_epoch = client_epoch
        self.fl_print = fl_print
        self.train_accuracy_mode = "mix"
        print(f"Train accuracy: {self.train_accuracy_mode} accuracy!")

    def mix_accuracy_train(self, model_weights, train_loader):
        self.model.load_state_dict(model_weights)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        scaler = GradScaler()
        epoch_loss = []
        for epoch in range(self.train_epoch):
            running_loss = []
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_loss.append(loss.item())
                if i % 50 == 49:  # 每50个小批量打印一次
                    average_loss = running_loss[i - 99:i + 1]
                    average_loss = sum(average_loss) / len(average_loss)
                    log_print(f'avg loss: {average_loss:.3f} in batch: {i + 1} at client epoch: {epoch + 1}',
                              self.fl_print)
                    epoch_loss.append(average_loss)

        return sum(epoch_loss) / len(epoch_loss)

    def full_accuracy_train(self, model_weights, train_loader):
        self.model.load_state_dict(model_weights)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train_size = len(train_loader.dataset)
        epoch_loss = []
        for epoch in range(self.train_epoch):
            running_loss = []
            running_corrects = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                running_loss.append(loss.item())
                running_corrects += torch.sum(preds == labels.data)
                if i % 50 == 49:  # 每50个小批量打印一次
                    average_loss = running_loss[i - 99:i + 1]
                    average_loss = sum(average_loss) / len(average_loss)
                    log_print(f'avg loss: {average_loss:.3f} in batch: {i + 1} at client epoch: {epoch + 1}',
                              self.fl_print)
                    epoch_loss.append(average_loss)
            epoch_acc = running_corrects.double() / train_size
            print(f"user train acc: {epoch_acc:.4f}")
        return  sum(epoch_loss) / len(epoch_loss), epoch_loss

    def shared_train_model(self, model_weights, train_loader):
        if self.train_accuracy_mode == "full":
            return  self.full_accuracy_train(model_weights, train_loader)
        elif self.train_accuracy_mode == "mix":
            return self.mix_accuracy_train(model_weights, train_loader)
        else:
            raise ValueError(f"train_accuracy_mode {self.train_accuracy_mode} not supported")

    def get_model_weights(self):
        return copy.deepcopy(self.model.state_dict())


