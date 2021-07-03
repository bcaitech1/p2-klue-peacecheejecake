import csv
import os
import matplotlib.pyplot as plt

from typing import Iterable


class CSVLogger():
    def __init__(self):
        self.train = []
        self.valid = []
        self.train_file = ''
        self.valid_file = ''

    def __call__(self, csvfiles: Iterable):
        self.train_file, self.valid_file = csvfiles
        self.load_from_csv()
        self.cleanup()

        return self.epochs()

    def load_from_csv(self):
        try:
            self.load_train_from_csv()
        except ValueError:
            print("WARNING: no train data loaded.")

        try:
            self.load_valid_from_csv()
        except ValueError:
            print("WARNING: no valid data loaded.")

        return self.epochs()
    
    def load_train_from_csv(self):
        if self.train:
            self.train = []

        if os.path.exists(self.train_file):
            with open(self.train_file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 3:
                        epoch, loss, accuracy = row
                        self.train.append([int(epoch), float(loss), float(accuracy)])
                    elif len(row) == 4:
                        epoch, loss, accuracy, _ = row
                        self.train.append([int(epoch), float(loss), float(accuracy)])
                    else:
                        raise ValueError("Wrong form: cannot load from csv file")
    
    def load_valid_from_csv(self):
        if self.valid:
            self.valid = []
            
        if os.path.exists(self.valid_file):
            with open(self.valid_file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 3:
                        epoch, loss, accuracy = row
                        self.valid.append([int(epoch), float(loss), float(accuracy)])
                    elif len(row) == 4:
                        epoch, loss, accuracy, state = row
                        self.valid.append([int(epoch), float(loss), float(accuracy), state])
                    else:
                        raise ValueError("Wrong form: cannot load from csv file")

    def save_to_csv(self):
        self.save_train_to_csv()
        self.save_valid_to_csv()

    def save_train_to_csv(self):
        if self.train_file:
            with open(self.train_file, 'w', newline='') as f:
                writer = csv.writer(f)
                for values in self.train:
                    writer.writerow(values)
        else:
            raise ValueError("Failed to write on train file.")

    def save_valid_to_csv(self):
        if self.valid_file:
            with open(self.valid_file, 'w', newline='') as f:
                writer = csv.writer(f)
                for values in self.valid:
                    writer.writerow(values)
        else:
            raise ValueError("Failed to write on valid file.")

    def log_last_train_to_csv(self):
        self.log_train_to_csv(*self.train[-1])
    
    def log_last_valid_to_csv(self):
        self.log_valid_to_csv(*self.valid[-1])

    def log_train_to_csv(self, *values):
        if self.train_file:
            with open(self.train_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(values)
        else:
            raise ValueError("Failed to write on train file.")
    
    def log_valid_to_csv(self, *values):
        if self.train_file:
            with open(self.valid_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(values)
        else:
            raise ValueError("Failed to write on valid file.")

    def recover(self, dest_epoch: int):
        self.recover_train(dest_epoch=dest_epoch)
        self.recover_valid(dest_epoch=dest_epoch)

    def recover_train(self, dest_epoch: int):
        '''train needs cleanup first.
        '''
        if dest_epoch > self.train[-1][0]:
            raise ValueError
        else:
            epochs = [logs[0] for logs in self.train]
            last_idx = epochs.index(dest_epoch)
            self.train = self.train[:last_idx + 1]

    def recover_valid(self, dest_epoch: int):
        '''valid needs cleanup first.
        '''
        if dest_epoch > self.valid[-1][0]:
            raise ValueError
        else:
            epochs = [logs[0] for logs in self.valid]
            last_idx = epochs.index(dest_epoch)
            self.valid = self.valid[:last_idx + 1]

    def __str__(self):
        str_ = f"({len(self.train)} train and {len(self.valid)} valid logs)"
        return str_

    def epochs(self):
        if self.train:
            return self.train[-1][0]
        else:
            return 0

    def log(self, train, valid):
        self.log_train(*train)
        self.log_valid(*valid)

    def log_train(self, *values):
        '''
        :values: [epoch, loss, accuracy]
        '''
        self.train.append(list(values))

    def log_valid(self, *values):
        '''
        :values: [epoch, loss, accuracy, checkpoint]
        '''
        self.valid.append(list(values))

    def delete_train_logs(self, num_to_del):
        self.train = self.train[:-num_to_del]

    def delete_valid_logs(self, num_to_del):
        self.train = self.train[:-num_to_del]

    def cleanup(self):
        epochs = []
        new_train = []
        deleted_train_logs = 0
        for epoch, loss, acc in self.train:
            if epoch in epochs:
                deleted_train_logs += 1
            else:
                epochs.append(epoch)
                new_train.append([epoch, loss, acc])
        
        epochs = []
        new_valid = []
        deleted_valid_logs = 0
        for epoch, loss, acc, state in self.valid:
            if epoch in epochs:
                deleted_valid_logs += 1
            else:
                epochs.append(epoch)
                new_valid.append([epoch, loss, acc, state])
        
        self.train = new_train
        self.valid = new_valid

        return deleted_train_logs, deleted_valid_logs

    def get_losses(self):
        train_losses = [x[1] for x in self.train]
        valid_losses = [x[1] for x in self.valid]
        return list(*zip(train_losses, valid_losses))
    
    def get_accuracies(self):
        train_accuracies = [x[2] for x in self.train]
        valid_accuracies = [x[2] for x in self.valid]
        return list(*zip(train_accuracies, valid_accuracies))

    def plot(self, suptitle=None):
        train_epoch, train_loss, train_acc = zip(*self.train)
        valid_epoch, valid_loss, valid_acc, _ = zip(*self.valid)

        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        
        axes[0].set_title("Loss")
        axes[1].set_title("Accuracy")

        train_color, valid_color = "#ff922b", "#22b8cf"
        
        axes[0].plot(train_epoch, train_loss, c=train_color, label="train")
        axes[0].plot(valid_epoch, valid_loss, c=valid_color, label="valid")
        axes[1].plot(train_epoch, train_acc,  c=train_color, label="train")
        axes[1].plot(valid_epoch, valid_acc,  c=valid_color, label="valid")

        for ax in axes:
            ax.legend()
            ax.set_xlabel("epoch")

        if suptitle:
            fig.suptitle(suptitle, fontsize=15)
            fig.tight_layout()

        plt.show()
