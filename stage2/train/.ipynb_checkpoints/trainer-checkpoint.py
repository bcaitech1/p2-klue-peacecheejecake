from utils import *
from data import *
from .config import ConfigTree

import csv
import json
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import __dict__ as torch_loss_dict

from adamp import AdamP, SGDP
import wandb



##########################################
# TRAINER ################################
##########################################


class Trainer():
    def __init__(self, config: ConfigTree):
        self.config = config
        self._configure()
        self._load_data_loader()
 

    def __call__(self):
        self._configure()
        self._load_data_loader()
        return self


    def _data_loader(self, dataset, shuffle):
        return DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size,
            shuffle=shuffle, 
            num_workers=self.config.system.num_workers, 
            pin_memory=True
        )


    def _load_data_loader(self):
        train_set, valid_set = train_valid_split(
            BasicDataset(
                glob(os.path.join(self.config.path.train, '*.jpg')),
                labeled=True,
                preprocess=self.config.data.preprocess,
                augment=False
            ),
            valid_ratio=self.config.data.valid_ratio,
            shuffle=False,
            valid_balanced=self.config.data.valid_balanced
        )
        
        test_set = BasicDataset(
            glob(os.path.join(self.config.path.test, '*.jpg')),
            labeled=False, 
            preprocess=self.config.data.preprocess,
            upscale=False
        )

        self.train_loader = self._data_loader(train_set, True)
        self.valid_loader = self._data_loader(valid_set, False)
        self.test_loader = self._data_loader(test_set, False)


    def _configure(self):
        self.device = self.config.system.device
        self.model = self.config.model.model.to(self.device)
        
        self.criterion = torch_loss_dict[self.config.train.loss.criterion]()
        
        self.logger = self.config.train.logger
        self.log_files = [os.path.join(self.config.path.logs, self.model.name + '_train_loss.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_train_acc.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_valid_loss.csv'),
                          os.path.join(self.config.path.logs, self.model.name + '_valid_acc.csv')]
        
        self.epochs = self.load_logs()
        
        if self.config.train.lr.scheduler:
            scheduler, kwarg = self.config.train.lr.scheduler, self.config.train.lr.scheduler_kwarg
        else:
            scheduler = None

        self._update_optimizer()
        
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **kwarg)
    

    def _update_optimizer(self):
        self.lr = self.config.train.lr.bas
        optim_name = self.config.train.optimizer.name.lower()

        if optim_name == 'adam':
            self.optimizer = optim.Adam(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'sgd':
            self.optimizer = optim.SGD(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                momentum=self.config.train.momentum,
                weight_decay=self.config.train.weight_decay,
                nesterov=self.config.train.nesterov
            )
        elif optim_name == 'adamp':
            self.optimizer = AdamP(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                betas=self.config.train.betas, 
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'sgdp':
            self.optimizer = SGDP(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.lr,
                weight_decay=self.config.train.weight_decay,
                momentum=self.config.train.momentum,
                nesterov=self.config.train.nesterov
            )
        else:
            raise NameError("Register proper optimizer if needed")
        if separate:
            self.lr = {
                    'backbone': self.config.train.lr.backbone,
                    'classifier': self.config.train.lr.classifier
                }

            params_backbone, params_classifier = [], []
            for module_name, module in self.model.named_children():
                for param in module.parameters():
                    if 'fc' in module_name and param.requires_grad:
                        params_classifier.append(param)
                    elif param.requires_grad:
                        params_backbone.append(param)

            if optim_name == 'adam':
                self.optimizer_backbone = optim.Adam(
                    params_backbone,
                    lr=self.config.train.lr.backbone,
                    weight_decay=self.config.train.weight_decay
                )
                self.optimizer_classifier = optim.Adam(
                    params_classifier,
                    lr=self.config.train.lr.classifier,
                    weight_decay=self.config.train.weight_decay
                )
            elif optim_name == 'sgd':
                self.optimizer_backbone = optim.SGD(
                    params_backbone,
                    lr=self.config.train.lr.backbone,
                    momentum=self.config.train.momentum,
                    weight_decay=self.config.train.weight_decay,
                    nesterov=self.config.train.nesterov
                )
                self.optimizer_classifier = optim.SGD(
                    params_classifier,
                    lr=self.config.train.lr.classifier,
                    momentum=self.config.train.momentum,
                    weight_decay=self.config.train.weight_decay,
                    nesterov=self.config.train.nesterov
                )
            elif self.config.train.optimizer.lower() == 'adamp':
                self.optimizer_backbone = AdamP(
                    params_backbone,
                    lr=self.lr['backbone'],
                    betas=self.config.train.betas, 
                    weight_decay=self.config.train.weight_decay
                )
                self.optimizer_classifier = AdamP(
                    params_classifier,
                    lr=self.lr['classifier'],
                    betas=self.config.train.betas, 
                    weight_decay=self.config.train.weight_decay
                )
            elif self.config.train.optimizer.lower == 'sgdp':
                self.optimizer_backbone = SGDP(
                    params_backbone,
                    lr=self.lr['backbone'],
                    weight_decay=self.config.train.weight_decay,
                    momentum=self.config.train.momentum,
                    nesterov=self.config.train.nesterov
                )
                self.optimizer_classifier = SGDP(
                    params_classifier,
                    lr=self.lr['backbone'],
                    weight_decay=self.config.train.weight_decay,
                    momentum=self.config.train.momentum,
                    nesterov=self.config.train.nesterov
                )
            else:
                raise NameError("Register proper optimizer if needed")


    def load_logs(self):
        dests = [self.logger.train['loss'], self.logger.train['acc'],
                 self.logger.valid['loss'], self.logger.valid['acc']]
        for i in range(4):
            if not os.path.exists(self.log_files[i]):
                continue

            with open(self.log_files[i], newline='') as csvfile:
                reader = csv.reader(csvfile)
                for epochs, value in reader:
                    dests[i].append((int(epochs), float(value)))

        if dests[0]:
            epochs_to_start = dests[0][-1][0]
        else:
            epochs_to_start = 0
        
        return epochs_to_start


    def write_log(self, dest: str, *values):
        with open(dest, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(values)

        
    def train_and_save(self):
        self._configure()

        print(f"[INFO] device={self.device}({torch.cuda.get_device_name(self.device)}), \
model={self.model.name}, epochs={self.config.train.num_epochs}")
        print(f"       lr={self.lr}, batch size={self.config.data.batch_size}, \
optimizer={self.config.train.optimizer}, weight decay={self.config.train.weight_decay}")
        if self.config.train.optimizer.name.lower() == 'adamp':
            print(f"       betas={self.config.train.betas}")
        elif self.config.train.optimizer.name.lower() in ('sgd', 'sgdp'):
            print(f"       momentum={self.config.train.momentum}, nesterov={self.config.train.nesterov}")
        print()
        print("Start of traning.")

        def save_with_name():
            # config_mode="json" with function to convert ConfigTree to JSON
            try:
                self.save(self.model.name, config_mode=None)
            except AttributeError:
                self.save(config_mode=None)


        for _ in range(self.config.train.num_epochs):
            valided = False
            saved   = False
            tested  = False
            plotted = False

            if self.config.train.valid_min == 0:
                self.valid()
            
            last_valid_acc = self.logger.valid['acc'][0][1] if self.logger.valid['acc'] else 0
            self.train_one_epoch()
            
            # k-fold validation
            if self.config.train.shuffle_period > 0 \
            and self.epochs % self.config.train.shuffle_period == 0:
                self.train_loader, self.valid_loader = self._train_valid_shuffle()

            # valid
            if self.config.train.valid_period > 0 \
            and self.epochs >= self.config.train.valid_min \
            and self.epochs % self.config.train.valid_period == 0:
                self.valid()
                valided = True
            
            # save
            if self.config.train.save_period > 0 \
            and self.epochs >= self.config.train.save_min \
            and self.epochs % self.config.train.save_period == 0 \
            and last_valid_acc >= self.config.train.save_min_acc:
                save_with_name()
                saved = True
            
            # test
            if self.config.train.test_period > 0 \
            and self.epochs >= self.config.train.test_min \
            and self.epochs % self.config.train.test_period == 0 \
            and last_valid_acc >= self.config.train.test_min_acc:
                self.infer_test_and_save()
                tested = True

            # plot loss & acc
            if self.config.train.plot_period > 0 \
            and self.config.train.plot_period > 0 \
            and self.epochs % self.config.train.plot_period == 0:
                self.logger.plot()
                plotted = True
                
        
        if not valided: self.valid()
        if not saved: save_with_name()
        if not tested: self.infer_test_and_save()
        if not plotted: self.logger.plot()

        print()
        print("End of training.")
        print()
        
        
    def train_one_epoch(self, add_loader=None):
        print(f"[Epoch {self.epochs + 1:03d}]", end="")
        
        self.model.train()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        starter.record()
        
        data_loader = add_loader if add_loader else self.train_loader
        epoch_time = train_loss = 0
        
        total = 0
        correct = 0
        false_neg = 0
        false_pos = 0
        
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            
            if self.config.train.optimizer.separate:
                self.optimizer_backbone.zero_grad()
                self.optimizer_classifier.zero_grad()
            else:
                self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            targets = targets.to(self.device)

            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.config.train.optimizer.separate:
                self.optimizer_backbone.step()
                self.optimizer_classifier.step()
                if self.config.train.lr.scheduler:
                    self.scheduler_backbone.step(loss)
                    self.scheduler_classifier.step(loss)
            else:
                self.optimizer.step()
                if self.config.train.lr.scheduler:
                    self.scheduler.step(loss)

            train_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

            ender.record()
            torch.cuda.synchronize()
            
            batch_time = starter.elapsed_time(ender) - epoch_time
            epoch_time += batch_time
            
            print(f'\r[Epoch {self.epochs + 1:03d}] (Batch #{batch:03d})  \
Loss: {train_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}  \
({time_str(epoch_time)})', end='')

        print()

        train_loss /= (batch + 1)
        accuracy  = correct / total

        self.epochs += 1
        
        self.logger.log_train(self.epochs, train_loss, accuracy)
        self.write_log(self.log_files[0], self.epochs, train_loss)
        self.write_log(self.log_files[1], self.epochs, accuracy)

        return train_loss, accuracy
        
        
    def valid(self):
        self._configure()

        valid_num = (self.epochs - self.config.train.valid_min + 1)\
         // self.config.train.valid_period if self.epochs > 0 else self.epochs
        print(f"[Valid {valid_num:03d}] ", end=" ")
        
        self.model.eval()
        
        valid_loss = 0
        correct = total = 0
        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(self.valid_loader):
                if self.config.train.valid_iters > 0 and batch >= self.config.train.valid_iters:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                valid_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1).detach()
                assert torch.all(predictions < 18), f"{torch.sum(predictions >= 18).item()}: out of bound"

                total += targets.shape[0]
                correct += predictions.eq(targets.detach()).sum().item()

                print(f'\r[Valid {valid_num:03d}] (Batch #{batch:03d})  \
Loss: {valid_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}', end='')

        print()
        valid_loss /= batch + 1
        accuracy = correct / total
        
        self.logger.log_valid(self.epochs, valid_loss, accuracy)
        self.write_log(self.log_files[2], self.epochs, valid_loss)
        self.write_log(self.log_files[3], self.epochs, accuracy)

        return valid_loss, accuracy
        
        
    def save(self, name='', postfix=None, config_mode=None):
        file_name = f'{datetime.today()}_{name}_{postfix if postfix else ""}'
        
        # save model parameter
        model_output = os.path.join(self.config.path.models, file_name + ".pth")
        torch.save(self.model.state_dict(), model_output)
        print(f"Saved model: {model_output}")

        # save configurations
        if config_mode == "json":
            config_output = os.path.join(self.config.path.configs, file_name + ".json")
            with open(config_output, "w") as f:
                json.dump(self.config, f)
                
                
    def infer_test_and_save(self):
        self._configure()
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()

        self.model.eval()
        indices = [path.split('/')[-1] for path in glob(os.path.join(self.config.path.test, '*.jpg'))]
        result = pd.DataFrame(columns=['ans'], index=indices)
        result.index.name = 'ImageID'
        
        with torch.no_grad():
            for i, (inputs, filenames) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                filenames = np.array(filenames)

                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1).detach()
                assert torch.all(predictions < 18), f"{torch.sum(predictions >= 18).item()}: out of bound"

                result['ans'][filenames] = predictions.cpu().numpy()

                ender.record()
                torch.cuda.synchronize()
                infer_time = starter.elapsed_time(ender)
                
                print(f"\rEvaluating: batch #{i} ({time_str(infer_time)})", end="")
        
        csv_name = f"{self.model.name}_{datetime.today()}.csv"
        csv_path = os.path.join(self.config.path.output, csv_name)
        result.to_csv(csv_path)
        
        print(f"\rSaved result: {csv_path}")