from ..modules import (
    models,
    loss, 
    functional as func
)
from . import lr_scheduler

from ..utils import (
    SEED,
    cuda_elpased_time_str,
    time_str,
    filename_from_datetime,
    empty_checkpoints,
    ConfusionMatrix,
)
from ..config import ConfigBranch
from ..data import (
    dataset,
    AutoTokenizer, 
    # add_special_tokens, 
    train_valid_split
)

import os
import math
from datetime import datetime
from time import time
from glob import glob
import shutil
import pickle
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adamp import AdamP, SGDP
from madgrad import MADGRAD
from transformers import AdamW

import neptune

from typing import Iterable



## TODO: create and separate :class TrainerConfig:
class Trainer:
    def __init__(self, config: ConfigBranch):
        self.config = config
        self._setup()
 

    def __call__(self):
        self._setup()
        return self
        

    def _setup(self):
        print("Start initial setup for trainer.")

        # device
        self.device = self.config.system.device

        # models
        ModelForTraining = getattr(models, self.config.model.arc)
        self.model = ModelForTraining(
            name=self.config.model.name,
            pretrained_id=self.config.model.pretrained_id,
            num_labels=self.config.data.num_classes, 
        ).to(self.device)
        
        if self.config.teacher.arc is not None:
            TeacherForTraining = getattr(models, self.config.model.teacher.arc)
            self.teacher = TeacherForTraining().to(self.device)
        else:
            self.teacher = None
        print("Loaded models.")
        
        # tokenizer
        self._load_tokenizer()
        print("Loaded tokenizer.")
        
        # data loaders
        self._load_data_loaders()
        print("Loaded data loaders.")
        
        # optimizer & lr scheduler
        self._update_optimizer()
        self._set_scheduler()

        # loss function
        self._set_loss_function()

        # load state dicts if requested
        if self.config.train.resuming_state:
            self.load_state_dicts()
        
        # init neptune for logging
        neptune.init(project_qualified_name='peace.cheejecake/stage2-KLUE')
        neptune.create_experiment(self.config.train.experiment_name)

        """LEGACY: manual logging
        """ 
        # self.logger = self.config.train.logger
        # self.log_files = [os.path.join(self.config.path.logs, self.model.name + '_train.csv'),
        #                   os.path.join(self.config.path.logs, self.model.name + '_valid.csv')]

        # valid_log_epochs = [logs[0] for logs in self.logger.valid]
        # valid_log_state_dicts = [logs[-1] for logs in self.logger.valid]

        # if self.config.train.resuming_state in valid_log_state_dicts:
        #     start_epoch = valid_log_epochs[valid_log_state_dicts.index(self.config.train.resuming_state)]
        #     self.logger.recover(start_epoch)
        #     self.logger.save_result()
        #     epoch = start_epoch
        # else:
        #     epoch = self.logger(self.log_files)

    
    def _reset_for_new_fold(self):
        # models
        ModelForTraining = getattr(models, self.config.model.arc)
        self.model = ModelForTraining(
            name=self.config.model.name,
            pretrained_id=self.config.model.pretrained_id,
            num_labels=self.config.data.num_classes
        ).to(self.device)
        
        if self.config.teacher.arc is not None:
            TeacherForTraining = getattr(models, self.config.model.teacher.arc)
            self.teacher = TeacherForTraining().to(self.device)
        else:
            self.teacher = None

        self._update_optimizer()
        self._set_scheduler()

        if self.config.train.resuming_state:
            self.load_state_dicts()

    
    def _update_optimizer(self):
        optim_name = self.config.train.optimizer.name.lower()
        lr = self.config.train.lr.base

        if optim_name == 'adam':
            self.optimizer = optim.Adam(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                betas=self.config.train.betas, 
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'sgd':
            self.optimizer = optim.SGD(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                momentum=self.config.train.momentum,
                weight_decay=self.config.train.weight_decay,
                nesterov=self.config.train.nesterov
            )
        elif optim_name == 'adamp':
            self.optimizer = AdamP(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                betas=self.config.train.betas, 
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'sgdp':
            self.optimizer = SGDP(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                weight_decay=self.config.train.weight_decay,
                momentum=self.config.train.momentum,
                nesterov=self.config.train.nesterov
            )
        elif optim_name == 'adamw':
            self.optimizer = AdamW(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                betas=self.config.train.betas, 
                weight_decay=self.config.train.weight_decay
            )
        elif optim_name == 'madgrad':
            self.optimizer = MADGRAD(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                weight_decay=self.config.train.weight_decay,
                momentum=self.config.train.momentum
            )
        else:
            raise NameError("Register proper optimizer if needed")


    def _set_scheduler(self):
        if self.config.train.lr.scheduler is None:
            self.scheduler = lr_scheduler.ConstantScheduler(
                optimizer=self.optimizer,
                lr=self.config.train.lr.base
            )
        else:
            SchedulerForTraining = getattr(lr_scheduler, self.config.train.lr.scheduler)
            if self.config.train.lr.warmup_steps is not None:
                warmup_steps = self.config.train.lr.warmup_steps
            else:
                warmup_steps = len(self.train_loaders[0].dataset) // self.config.data.batch_size

            if self.config.train.lr.min is not None:
                min_lr = self.config.train.lr.min
            else:
                min_lr = self.config.train.lr.base / 50

            if self.config.train.lr.cycle_steps is not None:
                cycle_steps = self.config.train.lr.cycle_steps
            else:
                cycle_steps = (
                    math.ceil(len(self.train_loaders[0].dataset) 
                    / self.config.data.batch_size) * self.config.train.num_epochs
                )

            self.scheduler = SchedulerForTraining(
                optimizer=self.optimizer,
                warmup_steps=warmup_steps,
                cycle_steps=cycle_steps,
                max_lr=self.config.train.lr.base,
                min_lr=min_lr,
                damping_ratio=self.config.train.lr.damping_ratio,
            )


    def _set_loss_function(self):
        try:
            LossForTraining = getattr(loss, self.config.train.criterion)
            self.criterion = LossForTraining()
        # except TypeError:
        #     self.criterion = getattr(loss.PresetLoss, self.config.train.criterion)
        except NameError:
            if self.config.train.criterion in loss.torch_loss_dict:
                self.criterion = loss.torch_loss_dict[self.config.train.criterion]()
            elif self.config.train.criterion == 'ArcFaceLoss':
                out_layer_in_features = list(self.model.parameters())[-2].size(1)
                out_layer_out_features = list(self.model.parameters())[-2].size(0)
                self.criterion = loss.AngularPenaltySMLoss(out_layer_in_features, out_layer_out_features)
            elif self.config.train.criterion == 'FocalLoss':
                pass
            elif self.config.train.criterion == 'LabelSmoothingLoss':
                pass
            elif self.config.train.criterion == 'F1Loss':
                pass
            else:
                raise ValueError    


    def _data_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size,
            shuffle=shuffle, 
            num_workers=self.config.system.num_workers, 
            pin_memory=True
        )


    def _load_data_loaders(self):
        DatasetForTraining = getattr(dataset, self.config.data.dataset_class)
        self.num_folds = int(self.config.data.num_folds)
        
        train_data = pd.read_csv(self.config.path.train)
        train_data, train_labels = (
            train_data.iloc[:, :8], train_data.iloc[:, 8].map(DatasetForTraining.label2idx)
        )

        self.train_loaders = []
        self.valid_loaders = []

        if self.config.path.valid is not None:
            train_set = DatasetForTraining(
                    data=train_data,
                    labels=train_labels,
                    tokenizer=self.tokenizer,
                    max_length=self.config.data.max_token_length,
                )
            valid_data = pd.read_csv(self.config.path.valid)
            valid_labels = valid_data['8'].map(DatasetForTraining.label2idx)
            valid_set = DatasetForTraining(
                data=valid_data,
                labels=valid_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_token_length,
            )
            train_loader = self._data_loader(train_set, shuffle=True)
            valid_loader = self._data_loader(valid_set, shuffle=False)

            self.train_loaders.append(train_loader)
            self.valid_loaders.append(valid_loader)

        elif self.num_folds >= 2:
            k_folder = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=SEED)
            for train_indices, valid_indices in k_folder.split(train_data, train_labels):
                train_set = DatasetForTraining(
                    data=train_data.iloc[train_indices],
                    labels=train_labels.iloc[train_indices],
                    tokenizer=self.tokenizer,
                    max_length=self.config.data.max_token_length,
                )
                valid_set = DatasetForTraining(
                    data=train_data.iloc[valid_indices],
                    labels=train_labels.iloc[valid_indices],
                    tokenizer=self.tokenizer,
                    max_length=self.config.data.max_token_length,
                )
                train_loader = self._data_loader(train_set, shuffle=True)
                valid_loader = self._data_loader(valid_set, shuffle=False)
                
                self.train_loaders.append(train_loader)
                self.valid_loaders.append(valid_loader)

        elif self.config.data.valid_ratio > 0:
            assert self.config.data.valid_ratio <= 1, "Not a valid valid ratio: (> 1)."
            
            train_set = DatasetForTraining(
                data=train_data,
                labels=train_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_token_length,
            )        
            train_set, valid_set = train_valid_split(
                dataset=train_set, 
                valid_ratio=self.config.data.valid_ratio, 
                shuffle=True
            )
            train_loader = self._data_loader(train_set, shuffle=True)
            valid_loader = self._data_loader(valid_set, shuffle=False)
            
            self.train_loaders.append(train_loader)
            self.valid_loaders.append(valid_loader)

        else:
            train_set = DatasetForTraining(
                data=train_data,
                labels=train_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_token_length,
            )        
            train_loader = self._data_loader(train_set, shuffle=True)
            
            self.train_loaders.append(train_loader)

        test_data = pd.read_csv(self.config.path.test)
        test_set = DatasetForTraining(
            data=test_data,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_token_length,
        )
        self.test_loader = self._data_loader(test_set, shuffle=False)


    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.data.tokenizer_class,
            self.config.model.pretrained_id
        )

        # if self.config.data.add_entity_tokens:
        #     add_special_tokens(
        #         tokenizer=self.tokenizer, 
        #         model=self.model.electra, 
        #         add_type='simple'
        #     )


    def load_state_dicts(self, resuming_state: str = None):
        r'''
        :resuming_state: prefix for files
        '''
        if resuming_state is None:
            resuming_state = self.config.train.resuming_state

        self.load_state_dict_to_model(resuming_state)
        if self.scheduler is not None:
            self.load_state_dict_to_scheduler(resuming_state)


    def load_state_dict_to_model(self, resuming_state: str = None):
        r'''
        :resuming_state: prefix for saved model
        '''
        if resuming_state is None:
            resuming_state = self.config.train.resuming_state

        if resuming_state:
            state_path = os.path.join(
                self.config.path.checkpoint, 
                resuming_state + '_model_.pth'
            )
            if os.path.exists(state_path):
                state_dict = torch.load(
                    state_path,
                    map_location=self.device
                )
                state_dict_keys = []
                for key in state_dict:
                    if key[:7] == 'module.':
                        key = '.'.join(key.split('.')[1:])
                    state_dict_keys.append(key)
                state_dict = dict(zip(state_dict_keys, state_dict.values()))
                
                self.model.load_state_dict(state_dict)
                print(f"Loaded state dict to model.")
                        
            else:
                print(f"WARNING: failed to load state dict for model - could not find the path.")


    def load_state_dict_to_scheduler(self, resuming_state: str):
        r'''
        :resuming_state: prefix for saved scheduler
        '''
        if resuming_state is None:
            resuming_state = self.config.train.resuming_state

        if self.config.train.resuming_state:
            state_path = os.path.join(
                self.config.path.checkpoint, 
                resuming_state + '_scheduler_.pkl'
            )
            if os.path.exists(state_path):
                with open(state_path, 'rb') as f:
                    state_dict = pickle.load(f)
                self.scheduler.load_state_dict(state_dict)
                print(f"Loaded state dict to scheduler.")
            else:
                print(f"WARNING: failed to load state dict for scheduler - could not find the path.")


    def _prepare_inputs(self, inputs):
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                inputs[key] = value.to(self.device)
        else:
            inputs = inputs.to(self.ldevice)
        return inputs


    def _train_info(self):
        print()
        print(f"[TRAINING INFO] experiment_name={self.config.train.experiment_name}")
        
        try:
            device_name = torch.cuda.get_device_name(self.device)
        except ValueError:
            device_name = ""
        print(f"device={self.device}({device_name})")

        valid_size = len(self.valid_loaders[0].dataset) if self.valid_loaders else 0
        print(
            f"data_size=({len(self.train_loaders[0].dataset)} + {valid_size}),",
            f"batch_size={self.config.data.batch_size}",
        )
        print(f"dataset={self.train_loaders[0].dataset.__class__.__name__}")
        print(f"arc={self.model.__class__.__name__}")

        print(f"optimizer={self.optimizer.__class__.__name__}")
        print(
            f"lr_base={self.config.train.lr.base},",
            f"scheduler={self.config.train.lr.scheduler},",
            f"weight_decay={self.config.train.weight_decay}",
            
            end=""
        )
        if self.config.train.optimizer.name.lower() in ('adam', 'adamp'):
            print(f", betas={self.config.train.betas}")
        elif self.config.train.optimizer.name.lower() in ('sgd', 'sgdp'):
            print(f", momentum={self.config.train.momentum}, nesterov={self.config.train.nesterov}")
        else:
            print()

        print()
        print(f">>> Start of traning ({self.num_folds} folds, {self.config.train.num_epochs} epochs).")


    def train_and_save(self):
        self._train_info()


        def save_with_name(postfix=None):
            try:
                name = f"fold{fold_idx}"
                saving_path = self.save_state_dicts(name, postfix=postfix)
            except AttributeError:
                print("WARNING: model has no name.")
                saving_path = self.save_state_dicts(postfix=postfix)
            
            return saving_path


        self.best_models = []
        for fold_idx in range(self.num_folds):
            self.train_loader = self.train_loaders[fold_idx]
            if self.valid_loaders:
                self.valid_loader = self.valid_loaders[fold_idx]

            print(f"---- FOLD {fold_idx} {'-' * 30}")
            if fold_idx > 0:
                self._reset_for_new_fold()

            best_in_fold = [0, '']
            not_saved_count = 0
            for epoch in range(self.config.train.num_epochs):
                ## TODO: implement early-stop -- by moving average
                if not_saved_count >= self.config.train.stop_count:
                    break

                epoch += 1
                train_loss, train_acc = self.train_one_epoch(epoch=epoch)
                neptune.log_metric(f'{self.config.train.experiment_name}_epoch_loss', train_loss)
                neptune.log_metric(f'{self.config.train.experiment_name}_epoch_acc', train_acc)
                    
                # valid
                if (
                    self.valid_loaders and
                    self.config.train.valid_period > 0 and
                    epoch >= self.config.train.valid_min_epoch and
                    epoch % self.config.train.valid_period == 0
                ):
                    valid_loss, valid_acc = self.valid(epoch=epoch)

                    neptune.log_metric(f'{self.config.train.experiment_name}_valid_loss', valid_loss)
                    neptune.log_metric(f'{self.config.train.experiment_name}_valid_acc', valid_acc)

                # save
                if (
                    self.config.train.save_period > 0 and
                    epoch >= self.config.train.save_min_epoch and
                    epoch % self.config.train.save_period == 0
                ):
                    if self.valid_loaders:
                        best_acc, best_model = best_in_fold
                        if best_acc < valid_acc:
                            self.remove_state_dicts(best_model)
                            best_in_fold = [valid_acc, save_with_name()]
                            not_saved_count = 0
                        else:
                            not_saved_count += 1
                    else:
                        save_with_name()
            
            # if fold_idx == 0:
            #     resuming_state = '2021-04-22-20_06_13.826937_fold0_'
            #     self.load_state_dicts(resuming_state=resuming_state)
            #     best_in_fold = [0.7572222222222222, resuming_state]
            #     not_saved_count = 0#
            #     for epoch in range(4, self.config.train.num_epochs):
            #         ## TODO: implement early-stop -- by moving average
            #         if not_saved_count >= self.config.train.stop_count:
            #             break

            #         epoch += 1
            #         train_loss, train_acc = self.train_one_epoch(epoch=epoch)
            #         neptune.log_metric(f'{self.config.train.experiment_name}_epoch_loss', train_loss)
            #         neptune.log_metric(f'{self.config.train.experiment_name}_epoch_acc', train_acc)
                        
            #         # valid
            #         if (
            #             self.valid_loaders and
            #             self.config.train.valid_period > 0 and
            #             epoch >= self.config.train.valid_min_epoch and
            #             epoch % self.config.train.valid_period == 0
            #         ):
            #             valid_loss, valid_acc = self.valid(epoch=epoch)

            #             neptune.log_metric(f'{self.config.train.experiment_name}_valid_loss', valid_loss)
            #             neptune.log_metric(f'{self.config.train.experiment_name}_valid_acc', valid_acc)

            #         # save
            #         if (
            #             self.config.train.save_period > 0 and
            #             epoch >= self.config.train.save_min_epoch and
            #             epoch % self.config.train.save_period == 0
            #         ):
            #             best_acc, best_model = best_in_fold
            #             if best_acc < valid_acc:
            #                 self.remove_state_dicts(best_model)
            #                 best_in_fold = [valid_acc, save_with_name()]
            #                 not_saved_count = 0
            #             else:
            #                 not_saved_count += 1
            # else:
            #     self._reset_for_new_fold()

            #     best_in_fold = [0, '']
            #     not_saved_count = 0
            #     for epoch in range(self.config.train.num_epochs):
            #         ## TODO: implement early-stop -- by moving average
            #         if not_saved_count >= self.config.train.stop_count:
            #             break

            #         epoch += 1
            #         train_loss, train_acc = self.train_one_epoch(epoch=epoch)
            #         neptune.log_metric(f'{self.config.train.experiment_name}_epoch_loss', train_loss)
            #         neptune.log_metric(f'{self.config.train.experiment_name}_epoch_acc', train_acc)
                        
            #         # valid
            #         if (
            #             self.valid_loaders and
            #             self.config.train.valid_period > 0 and
            #             epoch >= self.config.train.valid_min_epoch and
            #             epoch % self.config.train.valid_period == 0
            #         ):
            #             valid_loss, valid_acc = self.valid(epoch=epoch)

            #             neptune.log_metric(f'{self.config.train.experiment_name}_valid_loss', valid_loss)
            #             neptune.log_metric(f'{self.config.train.experiment_name}_valid_acc', valid_acc)

            #         # save
            #         if (
            #             self.config.train.save_period > 0 and
            #             epoch >= self.config.train.save_min_epoch and
            #             epoch % self.config.train.save_period == 0
            #         ):
            #             best_acc, best_model = best_in_fold
            #             if best_acc < valid_acc:
            #                 self.remove_state_dicts(best_model)
            #                 best_in_fold = [valid_acc, save_with_name()]
            #                 not_saved_count = 0
            #             else:
            #                 not_saved_count += 1

            if self.valid_loaders:
                self.best_models.append(best_in_fold[1])
                print(f"Best accuracy: {best_in_fold[1]}")
                print()
            else:
                save_with_name()

        print()
        print(f">>> End of training.")
        print()

        # self.empty_checkpoints_except_bests()

        
    def train_one_epoch(self, epoch: int = 0, add_loader: DataLoader = None):
        print(f"[Epoch {epoch:03d}]", end="")
        
        self.model.train()
        
        if self.device == torch.device("cpu"):
            starter = time.time()
        else:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        
        train_loader = add_loader if add_loader else self.train_loader
        epoch_time = epoch_loss = 0
        
        total = 0
        correct = 0
        confusion_matrix = ConfusionMatrix(self.config.data.num_classes)
        
        step = (epoch - 1) * self.config.data.batch_size
        
        for step_idx, (inputs, targets) in enumerate(train_loader):
            inputs = self._prepare_inputs(inputs)
            targets = targets.to(self.device)
            
            outputs = self.model(**inputs)
            
            if self.teacher:
                teacher_outputs = self.teacher(**inputs)
                loss = self.criterion(outputs, teacher_outputs, targets)
            else:
                loss = self.criterion(outputs, targets)
                # loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            
            """LEGACY: manual lr scheduling -- by moving average
            """
            # if self.config.lr.manual_scheduler:
            #     few = self.scheduler.averaging_few
            #     last_few_losses = [logs[1] for logs in self.logger.valid[-few:]]
            #     last_last_few_losses = [logs[1] for logs in self.logger.valid[-few-1:-1]]
            #     if last_last_few_losses and  np.mean(last_last_few_losses) < np.mean(last_few_losses):
            #         old_lr = self.lr
            #         self.lr /= self.scheduler.divider
            #         self.config.train.lr.base = self.lr
            #         print(f">>>>> Learning rate: {old_lr} -> {self.lr}")


            _, predictions = torch.max(outputs, dim=1)  #
            
            batch_loss = loss.item()
            batch_correct = predictions.eq(targets).sum().item()
            batch_accuracy = batch_correct / targets.shape[0]

            epoch_loss += batch_loss
            correct += batch_correct
            total += targets.shape[0]
            confusion_matrix(targets, predictions)

            if self.device == torch.device("cpu"):
                epoch_time = time() - starter
                epoch_time_str = time_str(epoch_time)
            else:
                ender.record()
                torch.cuda.synchronize()
                epoch_time = starter.elapsed_time(ender)
                epoch_time_str = cuda_elpased_time_str(epoch_time)
                
            print(
                f'\r[Epoch {epoch:03d}] (Step {step_idx:03d}) ({self.scheduler.get_lr():.5e})',
                f'Loss: {epoch_loss / (step_idx + 1):.5f},  Acc: {correct / total * 100:.3f}',
                f'({epoch_time_str})',
                
                end=''
            )

            # logging
            step += step_idx
            neptune.log_metric(f'{self.config.train.experiment_name}_batch_loss', batch_loss)
            neptune.log_metric(f'{self.config.train.experiment_name}_batch_acc', batch_accuracy)
            neptune.log_metric(f'{self.config.train.experiment_name}_lr', self.scheduler.get_lr())


        epoch_loss /= (step_idx + 1)
        epoch_accuracy  = correct / total
        f1_score = confusion_matrix.f1_score()

        # self.logger.log_train(epoch, epoch_loss, accuracy)
        # self.logger.log_last_train_to_csv()
        
        print()
        # print(
        #     f'----------- ',
        #     f'Loss: {epoch_loss:.5f},  Acc: {epoch_accuracy * 100:.3f},  F1 Score: {f1_score:.5f}',
        # )

        return epoch_loss, epoch_accuracy

        
    def valid(self, epoch: int = None):
        epoch = f" {epoch:03d}" if epoch is not None else ""
        print(f"[Valid{epoch}] ", end="")
        
        self.model.eval()
        
        valid_loss = 0
        correct = total = 0
        confusion_matrix = ConfusionMatrix(self.config.data.num_classes)

        try:
            valid_loader = self.valid_loader
        except AttributeError:
            valid_loader = self.valid_loaders[0]

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(valid_loader):
                inputs = self._prepare_inputs(inputs)
                targets = targets.to(self.device)
                outputs = self.model(**inputs)

                if self.teacher:
                    teacher_outputs = self.teacher(**inputs)
                    loss = self.criterion(outputs, teacher_outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                
                valid_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)

                assert torch.all(predictions < self.config.data.num_classes), (
                    f"{torch.sum(predictions >= self.config.data.num_classes).item()}"
                    f"/{self.config.data.num_classes}: out of bound"
                )

                total += targets.shape[0]
                correct += predictions.eq(targets).sum().item()
                confusion_matrix(targets, predictions)

                print(
                    f"\r[Valid{epoch}] (Batch #{batch:03d})",
                    f"Loss: {valid_loss / (batch + 1):.5f},  Acc: {correct / total * 100:.3f}", 

                    end=""
                )

        valid_loss /= batch + 1
        accuracy = correct / total
        f1_score = confusion_matrix.f1_score()
        
        print(
            f'\r[Valid{epoch}]',
            f'Loss: {valid_loss:.5f},  Acc: {accuracy * 100:.3f},  F1 Score: {f1_score:.5f}'
        )

        return valid_loss, accuracy
        
        
    def save_state_dicts(self, name="", postfix=None):
        file_name = f'{filename_from_datetime(datetime.today())}_{name}_{postfix if postfix else ""}'
        
        # save model
        model_out_path = os.path.join(self.config.path.checkpoint, file_name + "_model_.pth")
        torch.save(self.model.state_dict(), model_out_path)
        print(f"Saved model: {model_out_path}")

        # save scheduler
        if self.scheduler is not None:
            scheduler_out_path = os.path.join(self.config.path.checkpoint, file_name + "_scheduler_.pkl")
            with open(scheduler_out_path, 'wb') as newfile:
                pickle.dump(self.scheduler.state_dict(), newfile, pickle.HIGHEST_PROTOCOL)
            print(f"Saved scheduler: {scheduler_out_path}")

        return file_name


    def infer(self):
        if self.device == torch.device("cpu"):
            starter = time()
        else:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            starter.record()

        self.model.eval()

        test_size = len(self.test_loader.dataset)

        total_batch_num = test_size / self.config.data.batch_size
        total_batch_num = math.ceil(total_batch_num)

        
        with torch.no_grad():
            if self.config.path.valid is None and self.num_folds > 1:
                logits = torch.zeros((test_size, self.config.data.num_classes), dtype=torch.float).to(self.device)
                for fold_idx, checkpoint in enumerate(self.best_models):
                    self.load_state_dict_to_model(resuming_state=checkpoint)
                    logits_in_fold = torch.tensor([], dtype=torch.float).to(self.device)
                    for i, inputs in enumerate(self.test_loader):
                        inputs = self._prepare_inputs(inputs)
                        outputs = self.model(**inputs)
                        logits_in_fold = torch.cat([logits_in_fold, outputs])

                        if self.device == torch.device("cpu"):
                            infer_time = starter - time()
                            infer_time_str = time_str(infer_time)
                        else:
                            ender.record()
                            torch.cuda.synchronize()
                            infer_time = starter.elapsed_time(ender)
                            infer_time_str = cuda_elpased_time_str(infer_time)

                        print(f"\rEvaluating fold {fold_idx}: {i} / {total_batch_num} ({infer_time_str})", end="")
                        
                    logits += logits_in_fold

                result = torch.argmax(logits, dim=1).detach()
                print()
                
            else:
                result = torch.tensor([], dtype=int).to(self.device)
                for i, inputs in enumerate(self.test_loader):
                    inputs = self._prepare_inputs(inputs)
                    outputs = self.model(**inputs)
                    predictions = torch.argmax(outputs, dim=1).detach()
                    result = torch.cat([result, predictions])

                    if self.device == torch.device("cpu"):
                        infer_time = starter - time()
                        infer_time_str = time_str(infer_time)
                    else:
                        ender.record()
                        torch.cuda.synchronize()
                        infer_time = starter.elapsed_time(ender)
                        infer_time_str = cuda_elpased_time_str(infer_time)
                    
                    print(f"\rEvaluating: {i} / {total_batch_num} ({cuda_elpased_time_str(infer_time)})", end="")

                print()

            assert torch.all(result < self.config.data.num_classes), \
                        f"{torch.sum(result >= self.config.data.num_classes).item()}/{self.config.data.num_classes}: out of bound"
                

        print(f"\r{self.config.train.experiment_name}: End of evaluation ({cuda_elpased_time_str(infer_time)})")

        result = pd.DataFrame(result.cpu().numpy(), columns=['pred'])
        
        return result

                
    def infer_and_save_result(self):
        result = self.infer()

        try:
            sub_num = int(sorted(glob('/opt/ml/output/submissions/*.csv'))[-1][-7:-4]) + 1
        except IndexError:
            sub_num = 0
        
        csv_name = f"/opt/ml/output/submissions/submission{sub_num:03d}.csv"
        csv_path = os.path.join(self.config.path.submit, csv_name)
        result.to_csv(csv_path, index=False)
        
        print(f"\rSaved result: {csv_path}")

        return csv_path


    def empty_checkpoints_except_bests(self):
        for fold_idx, out_name in enumerate(self.best_models):
            checkpoint_path_base = os.path.join(self.config.path.checkpoint, out_name)
            model_path = checkpoint_path_base + '_model_.pth'
            scheduler_path = checkpoint_path_base + '_scheduler_.pkl'

            tmp_path = os.path.join(self.config.path.checkpoint, 'tmp')
            model_dest = os.path.join(tmp_path, f"fold_{fold_idx}_model_.pth")
            scheduler_dest = os.path.join(tmp_path, f"fold_{fold_idx}_scheduler_.pkl")

            shutil.move(model_path, model_dest)
            shutil.move(scheduler_path, scheduler_dest)

        empty_checkpoints()
        
        print("Rearanged files for evaluation.")


    def remove_state_dicts(self, filename):
        filepath_base = os.path.join(self.config.path.checkpoint, filename)
        
        model_path = filepath_base + '_model_.pth'
        scheduler_path = filepath_base + '_scheduler_.pkl'
        
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed model: {model_path}")
        if os.path.exists(scheduler_path):
            os.remove(scheduler_path)
            print(f"Removed scheduler: {scheduler_path}")



    # def infer_with_simple_tta(self, augments: Iterable = None):
    #     if not augments: augments = SimpleTTA.augments

    #     self.model.eval()

    #     test_size = len(self.test_loader.dataset)
    #     batch_size = self.config.data.batch_size
    #     total_batch_num = math.ceil(test_size / batch_size)
        
    #     with torch.no_grad():
    #         logits_all = torch.zeros((test_size, self.config.data.num_classes), dtype=torch.float).to(self.device)
    #         for aug_idx, augment in enumerate(augments):
    #             test_loader = self._data_loader(
    #                 dataset=SimpleTTADataset(self.test_loader.dataset.data, augment),
    #                 shuffle=False
    #             )

    #             logits = torch.zeros((test_size, self.config.data.num_classes), dtype=torch.float).to(self.device)
    #             for batch_idx, (inputs, _) in enumerate(test_loader):
    #                 inputs = inputs.to(self.device)
    #                 outputs = F.log_softmax(self.model(inputs), dim=1)
    #                 logits[batch_idx * batch_size: (batch_idx + 1) * batch_size] = outputs
                    
    #                 print(f"\rEvaluating #{aug_idx + 1}/{len(augments)} ({batch_idx + 1}/{total_batch_num})", end="")

    #             logits_all += logits.detach()

    #     print()
    #     print(f"End of evaluation.")


    #     indices = [path.split('/')[-1] for path in self.test_loader.dataset.data]
    #     result = pd.DataFrame(columns=['ans'], index=indices)
    #     result.index.name = 'ImageID'
    #     result.ans = torch.argmax(logits_all, dim=1).cpu().numpy()

    #     info_file = pd.read_csv(
    #         os.path.join(self.config.path.output, 'info.csv'),
    #         index_col='ImageID'
    #     )
    #     result = result.loc[info_file.index]
        
    #     return result



def ensemble_and_infer_test(
    trainer: Trainer, 
    models: Iterable[models.BasicModel], 
    weighted: bool=False
):
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    starter.record()
    
    test_data = trainer.test_loader.dataset.data
    test_size = len(test_data)
    num_classes = trainer.test_loader.dataset.num_classes
    
    config = trainer.config
    device = config.system.device
    test_path = config.path.test
    save_path = config.path.output

    model_in_trainer = models.DICT[config.model.arc]

    logits = torch.zeros((test_size, num_classes)).to(device)
    
    
    for model_idx, model in enumerate(models):
        if model_idx > 0:
            print(f"\rEvaluating: {model_idx}/{len(models)} ({cuda_elpased_time_str(elapsed_time)})", end="")
        else:
            print(f"\rEvaluating: {model_idx}/{len(models)}", end="")

        config.model.arc = model.__class__.__name__
        logits += trainer.logits()

        ender.record()
        torch.cuda.synchronize()
        elapsed_time = starter.elapsed_time(ender)

    predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()

    filenames = [path.split('/')[-1] for path in test_data]
    result = pd.DataFrame(index=filenames, columns=['ans'])
    result.index.name = 'ImageID'
    result.ans = predictions
    
    info = pd.read_csv(os.path.join(test_path, '..', 'info.csv'), index_col='ImageID')
    result = result.loc[info.index]

    weighted_symbol = 'w' if weighted else 'x'
    csv_name = f"ensemble_{len(models)}{weighted_symbol}_{filename_from_datetime(datetime.today())}.csv"
    csv_path = os.path.join(save_path, csv_name)
        
    result.to_csv(csv_path)

    print(f"\rSaved result: {csv_path}")

    config.model.arc = model_in_trainer.__class__.__name__
