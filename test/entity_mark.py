import sys
sys.path.append('/opt/ml/code/stage2')

import stage2
from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
dataset = stage2.data.EntityMarkedDatasetForElectra('/opt/ml/input/data/train/train.csv', tokenizer, True)


print(tokenizer.decode(dataset.tokenized_data.input_ids[0]))

print("Done.")