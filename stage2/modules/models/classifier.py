from .base import BasicModel

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiheadClassifier(BasicModel):
    def __init__(self, in_features: int, label_weight: int = 0.5, out_features: int = 18, name='classifier'):
        super(MultiheadClassifier, self).__init__(name)

        self.in_features = in_features
        self.out_features = out_features

        self.label_weight = label_weight

        self.age_classifier = nn.Linear(in_features, 3)
        self.gender_classifier = nn.Linear(in_features, 2)
        self.mask_classifier = nn.Linear(in_features, 3)

        self.label_classifier = nn.Linear(in_features, out_features)

    
    def _forward_impl(self, x):
        age = self.age_classifier(x)
        gender = self.gender_classifier(x)
        mask = self.mask_classifier(x)

        age = F.log_softmax(age, dim=1)
        gender = F.log_softmax(gender, dim=1)
        mask = F.log_softmax(mask, dim=1)

        label = self.label_classifier(x)
        label = F.log_softmax(label, dim=1)

        label_from_concat = torch.zeros_like(label, dtype=x.dtype).to(x.device)

        age_idx = [0, 1, 2] * 6
        gender_idx = ([0] * 3 + [1] * 3) * 3
        mask_idx = [0] * 6 + [1] * 6 + [2] * 6
        
        label_from_concat += age[:, age_idx] + gender[:, gender_idx] + mask[:, mask_idx]

        x = self.label_weight * label + (1 - self.label_weight) * label_from_concat

        return x


