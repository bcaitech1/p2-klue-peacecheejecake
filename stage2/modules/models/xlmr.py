from .base import BasicModel

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig


class BasicXLMRobertaForSequenceClassification(BasicModel):
    def __init__(
        self, 
        name: str = 'xlm-roberta',
        model_config = None,
        pretrained_id: str = "xlm-roberta-base",
        num_labels: int = 42,
        dropout_rate: float = 0.5,
    ):
        super(BasicXLMRobertaForSequenceClassification, self).__init__(name=name)

        if model_config is None:
            model_config = XLMRobertaConfig.from_pretrained(pretrained_id)
        self.hidden_size = model_config.hidden_size

        if pretrained_id:
            self.roberta = XLMRobertaModel.from_pretrained(pretrained_id)
            self.classifier = nn.Sequential(OrderedDict({
                'dense': nn.Linear(self.hidden_size, self.hidden_size),
                'dropout': nn.Dropout(dropout_rate),
                'relu': nn.ReLU(),
                'out_proj': nn.Linear(self.hidden_size, num_labels),
            }))
        else:
            raise NotImplementedError
        
        self.num_labels = num_labels

    
    def forward(self, **inputs):
        x = self.roberta(**inputs)
        x = x.last_hidden_state[:, 0, :]
        x = self.classifier(x)
        return x



class XLMRobertaForPreMarkedSequenceClassification(BasicXLMRobertaForSequenceClassification):
    def __init__(
        self, 
        name: str = 'xlm-roberta',
        model_config = None,
        pretrained_id: str = "xlm-roberta-base",
        num_labels: int = 42,
        dropout_rate: float = 0.5,
    ):
        super(XLMRobertaForPreMarkedSequenceClassification, self).__init__(
            name = name,
            model_config = model_config,
            pretrained_id = pretrained_id,
            num_labels = num_labels,
            dropout_rate = dropout_rate,
        )
        self.entity_classifier = nn.Sequential(OrderedDict({
                'dense': nn.Linear(self.hidden_size * 2, self.hidden_size),
                'dropout': nn.Dropout(dropout_rate),
                'relu': nn.ReLU(),
                'out_proj': nn.Linear(self.hidden_size, num_labels),
        }))

    def forward(self, **inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        last_head1_indices = inputs['last_head1_indices']
        last_head2_indices = inputs['last_head2_indices']

        x = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
        ).last_hidden_state

        x1 = x[list(range(x.size(0))), last_head1_indices, :]
        x2 = x[list(range(x.size(0))), last_head2_indices, :]
        x = torch.cat([x1, x2], dim=1)
        x = self.entity_classifier(x)
        return x


class XLMRobertaForPreMarkedSequenceConcatClassification(XLMRobertaForPreMarkedSequenceClassification):
    def forward(self, **inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        last_head1_indices = inputs['last_head1_indices']
        last_head2_indices = inputs['last_head2_indices']

        x = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
        ).last_hidden_state

        x0 = x[:, 0, :]
        x1 = x[list(range(x.size(0))), last_head1_indices, :]
        x2 = x[list(range(x.size(0))), last_head2_indices, :]

        x = torch.cat([x1, x2], dim=1)
        x = self.entity_classifier(x)

        x0 = self.classifier(x0)
        x += x0

        return x
