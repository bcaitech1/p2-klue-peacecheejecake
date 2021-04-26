import torch
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config, GPT2Tokenizer

device = torch.device("cuda:0")

config = GPT2Config()
model = GPT2Model(config)
lmhead = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")

print(lmhead)

kogpt2qg = torch.load('/opt/ml/etc/QG_kogpt2.pth', map_location=device)