from .base import BasicModel

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraConfig


class BasicElectraForSequenceClassification(BasicModel):
    def __init__(
        self, 
        name: str = 'ko-electra',
        model_config = None,
        pretrained_id: str = "monologg/koelectra-base-v3-discriminator",
        num_labels: int = 42,
        dropout_rate: float = 0.5,
    ):
        super(BasicElectraForSequenceClassification, self).__init__(name=name)

        if model_config is None:
            model_config = ElectraConfig.from_pretrained(pretrained_id)
        self.embedding_size = model_config.embedding_size
        self.hidden_size = model_config.hidden_size

        if pretrained_id:
            self.electra = ElectraModel.from_pretrained(pretrained_id)
            self.classifier = nn.Sequential(OrderedDict({
                'dense': nn.Linear(self.embedding_size, self.hidden_size),
                'dropout': nn.Dropout(dropout_rate),
                'out_proj': nn.Linear(self.hidden_size, num_labels)
            }))
        else:
            raise NotImplementedError
        
        self.num_labels = num_labels
    
    def forward(self, **inputs):
        x = self.electra(**inputs)
        x = x.last_hidden_state[:, 0, :]
        x = self.classifier(x)
        return x


class ElectraForPreMarkedSequenceClassification(BasicElectraForSequenceClassification):
    def __init__(
        self, 
        name: str = 'ko-electra',
        model_config = None,
        pretrained_id: str = "monologg/koelectra-base-v3-discriminator",
        num_labels: int = 42,
        dropout_rate: float = 0.5,
    ):
        super(ElectraForPreMarkedSequenceClassification, self).__init__(
            name = name,
            model_config = model_config,
            pretrained_id = pretrained_id,
            num_labels = num_labels,
            dropout_rate = dropout_rate,
        )
        self.entity_classifier = nn.Sequential(OrderedDict({
                'dense': nn.Linear(self.embedding_size * 2, self.hidden_size),
                'dropout': nn.Dropout(dropout_rate),
                'out_proj': nn.Linear(self.hidden_size, num_labels)
        }))

    def forward(self, **inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        last_head1_indices = inputs['last_head1_indices']
        last_head2_indices = inputs['last_head2_indices']

        x = self.electra(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
        ).last_hidden_state

        x1 = x[list(range(x.size(0))), last_head1_indices, :]
        x2 = x[list(range(x.size(0))), last_head2_indices, :]
        x = torch.cat([x1, x2], dim=1)
        x = self.entity_classifier(x)
        return x


class ElectraForPreMarkedSequenceConcatClassification(ElectraForPreMarkedSequenceClassification):
    def forward(self, **inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        last_head1_indices = inputs['last_head1_indices']
        last_head2_indices = inputs['last_head2_indices']

        x = self.electra(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
        ).last_hidden_state

        x0 = x[:, 0, :]
        x1 = x[list(range(x.size(0))), last_head1_indices, :]
        x2 = x[list(range(x.size(0))), last_head2_indices, :]

        x = torch.cat([x1, x2], dim=1)
        x = self.entity_classifier(x)

        x0 = self.classifier(x0)
        x += x0

        return x
