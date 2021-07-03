import torch.nn as nn
from typing import Iterable


class BasicModel(nn.Module):
    def __init__(self, name='model'):
        super(BasicModel, self).__init__()
        self.name = name
        
    def init(self):
        self.__init__()
        return self

    def requires_grad(self, mode=True):
        for parameter in self.parameters():
            parameter.requires_grad = mode

    def requires_grad(self, parameters: Iterable[nn.Parameter], mode=True):
        for parameter in parameters:
            parameter.requires_grad = mode

    def __str__(self):
        return self.__class__.__name__


class BasicSequential(BasicModel):
    def __init__(self, name='seq'):
        super(BasicSequential, self).__init__(name)

    def forward(self, x):
        for child in self.children():
            x = child(x)
        
        return x


