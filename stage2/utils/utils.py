import os
from datetime import datetime
from glob import glob
import numpy as np

# import scipy.stats

import torch
import torch.nn as nn

from .seed import *


##########################################
# UTILITIES ##############################
##########################################

def datetime_str(datetime: datetime, include_date=False, include_time=True, include_decimal=False):
    '''Convert datetime object to str
    '''
    date, time = str(datetime).split()
    time, decimal = time.split('.')
    datetime_str = ''
    if include_date: datetime_str += date
    if include_time: datetime_str += f' {time}'
    if include_decimal: datetime_str += f'.{decimal[:3]}'
    return datetime_str


def cuda_elpased_time_str(millisecond: float, include_decimal=False):
    r'''Convert time to str
    Args:
        millisecond (float): elapsed time recorded by torch.cuda.Event
        include_decimal (bool): whether include decimal points to second
    '''
    second, decimal = divmod(int(millisecond), 1000)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    decimal = str(decimal).rjust(3, '0')

    time_str = f'{minute:02d}:{second:02d}'
    if hour > 0:
        time_str = f'{hour:02d}:' + time_str

    if include_decimal:
        time_str += f'.{decimal}'
    
    return time_str


def time_str(second: float):
    r'''Convert time to str
    Args:
        second (float): elapsed time recorded by time
        include_decimal (bool): whether include decimal points to second
    '''
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)

    time_str = f'{minute:02d}:{second:02d}'
    if hour > 0:
        time_str = f'{hour:02d}:' + time_str

    return time_str


def filename_from_datetime(datetime: datetime):
    '''Create filename from datetime
    '''
    filename = '_'.join(str(datetime).split(':'))
    filename = '-'.join(filename.split())
    
    return filename


def empty_logs():
    for log in glob('/opt/ml/output/logs/*.csv'):
        os.remove(log)


def empty_checkpoints():
    for model in glob('/opt/ml/output/checkpoints/2021*'):
        os.remove(model)


try:
    load_state_dict_from_url = torch.hub.load_state_dict_from_url
except:
    load_state_dict_from_url = torch.utils.model_zoo.load_url




class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._reset()


    def __call__(self, targets: torch.Tensor, predictions: torch.Tensor):
        assert targets.shape == predictions.shape
        
        input_size = targets.size(0)
        new_info = self._new_matrix()
        for input_idx in range(input_size):
            target = targets[input_idx].item()
            prediction = predictions[input_idx].item()
            if target == prediction:
                new_info[target, prediction] += 1
            else:
                new_info[target, prediction] += 1
                new_info[prediction, target] += 1

        self.matrix += new_info
                

    def __add__(self, value):
        if not isinstance(value, ConfusionMatrix):
            raise TypeError("Only ConfusionMatrix can be added to ConfusionMatrix.")

        self.matrix += value.matrix

        return self
    

    def _reset(self):
        self.matrix = self._new_matrix()

    
    def _new_matrix(self):
        return torch.zeros((self.num_classes, self.num_classes), dtype=torch.float)


    def recall(self):
        return (self.matrix / self.matrix.sum(dim=0)).diagonal().mean().item()


    def precision(self):
        return (self.matrix / self.matrix.sum(dim=1)).diagonal().mean().item()

    
    def f1_score(self):
        # return hmean([self.recall(), self.precision()])
        return 2 / (1 / self.recall() + 1 / self.precision())
