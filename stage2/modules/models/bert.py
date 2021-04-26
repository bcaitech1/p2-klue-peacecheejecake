from .base import BasicModel

import torch.nn as nn
from transformers import BertForSequenceClassification


# class VanillaBert(BasicModel):
#     def __init__(
#         self,
#         num_labels: int = 42,
#         pooler_idx: int = 0,
#         name: str = 'vanillabert',
#     ):
#         super(VanillaBert, self).__init__(name)
#         # BERT로부터 얻은 128(=max_length)개 hidden state 중 몇 번째를 활용할 지 결정. Default - 0(CLS 토큰의 인덱스)
#         self.idx = 0 if pooler_idx == 0 else pooler_idx
#         self.backbone = BertForSequenceClassification.from_pretrained(
#             "bert-base-multilingual-cased"
#         ).bert
#         self.layernorm = nn.LayerNorm(768)  # 768: output length of backbone, BERT
#         self.dropout = nn.Dropout()
#         self.relu = nn.ReLU()
#         self.linear = nn.Linear(in_features=768, out_features=num_labels)

#     def forward(self, **inputs):
#         x = self.backbone(**inputs)

#         # backbone으로부터 얻은 128(토큰 수)개 hidden state 중 어떤 것을 활용할 지 결정. Default - 0(CLS 토큰)
#         x = x.last_hidden_state[:, self.idx, :] 
#         x = self.layernorm(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         output = self.linear(x)
#         return output
