import sys
sys.path.append('/opt/ml/code/stage2')

import stage2
import pandas as pd
import numpy as np
import pickle


def preprocess(data):
    labels = data['8'].map(label_types)

    return pd.DataFrame({
        'sentence': data['1'],
        'entity_01': data['2'],
        'entity_02': data['5'],
        'labels': labels,
    })


def tokenize(data, tokenizer):
    entities = []
    sentences = []
    for idx, (entity_01, entity_02) in enumerate(zip(data['entity_01'], data['entity_02'])):
        sentences.append(data['sentence'].iloc[idx])
        concat_result = ''
        if type(entity_01) != str:
            entity_01 = '[NAN]'
        if type(entity_02) != str:
            entity_02 = '[NAN]'
        concat_result = entity_01 + '[SEP]' + entity_02
        entities.append(concat_result)

    return tokenizer(
        text=entities,
        text_pair=sentences,
        return_tensors='pt',
        padding=True,
        max_length=100,
        add_special_tokens=True,
    )


with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_types = pickle.load(f)
train_raw = pd.read_csv('/opt/ml/input/data/train/train+all.tsv')
print(f"TRAIN SIZE: {len(train_raw)}")


sample = train_raw.iloc[np.random.randint(0, len(train_raw), 5000)]
sample = preprocess(sample)
# print(f"TRAIN SAMPLE: {sample}")

tokenizer = stage2.data.AutoTokenizer.from_pretrained(
    "ElectraTokenizer",
    "monologg/koelectra-base-v3-discriminator",
)
tokenized_sample = tokenize(sample, tokenizer)
# print(tokenized_sample)

model = stage2.models.BasicElectraForSequenceClassification()

sample_outputs = model(**tokenized_sample)
print(sample_outputs.shape)

# train_set = stage2.data.BasicDatasetForElectra(
#     data_path='/opt/ml/input/data/train/train+all.tsv',
#     labeled=True,
#     tokenizer=tokenizer,
# )