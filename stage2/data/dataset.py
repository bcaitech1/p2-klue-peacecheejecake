from ..utils import *
from .functional import *

from collections import Counter
import pandas as pd
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer


##########################################
# DATASET ################################
##########################################   
        

# class BasicDatasetForTransformers(Dataset):
#     def __init__(
#         self, 
#         data_path: str, 
#         labeled: bool
#     ):
#         super(BasicDatasetForTransformers, self).__init__()
#         self.data_path = data_path
#         self.labeled = labeled
#         self.num_classes = 42
        
        
#     def __getitem__(self, idx: int):
#         raise NotImplementedError
    
    
#     def __len__(self):
#         return len(self.data)



class BasicDatasetForElectra(Dataset):
    num_classes = 42

    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label2idx = pickle.load(f)


    def __init__(
        self, 
        data: pd.DataFrame, 
        labels: pd.Series, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        super(BasicDatasetForElectra, self).__init__()

        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        self.tokenize()


    def __getitem__(self, idx: int):
        inputs = {key: value[idx].clone() for key, value in self.tokenized_data.items()}
        if self.labels is not None:
            label = self.labels.iloc[idx]
            label = torch.tensor(label)
            return inputs, label
        else:
            return inputs
   
    
    def tokenize(self):
        entities = []
        sentences = []
        for idx in range(len(self.data)):
            sentence, entity1, entity2 = self.data.iloc[idx, [1, 2, 5]]
            if type(entity1) != str:
                entity1 = '[NAN]'
            if type(entity2) != str:
                entity2 = '[NAN]'
            sentences.append(sentence)
            entities.append(entity1 + '[SEP]' + entity2)

        self.tokenized_data = self.tokenizer(
            text=entities,
            text_pair=sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )


    def __len__(self):
        return len(self.data)
        

class EntityMarkedDatasetForElectra(BasicDatasetForElectra):     
    def tokenize(self):
        # entities = []
        marked_sentences = []
        for idx in range(len(self.data)):
            sentence, entity1, e1s, e1e, entity2, e2s, e2e = self.data.iloc[idx, 1:8]
            if e1s < e2s:
                sentence = sentence[:e1s] + f'❗{entity1}✅' \
                     + sentence[e1e + 1:] + f'✨{entity2}⭐' + sentence[e2e: + 1:]
            else:
                sentence = sentence[:e2s] + f'✨{entity2}⭐' \
                     + sentence[e2e + 1:] + f'❗{entity1}✅' + sentence[e1e + 1:]
            marked_sentences.append(sentence)
            # entities.append(entity1 + '[SEP]' + entity2)

        self.tokenized_data = self.tokenizer(
            text=marked_sentences,
            # text_pair=marked_sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )


class EntityPreMarkedDatasetForElectra(BasicDatasetForElectra):
    def tokenize(self):
        # entity_tokens = ["❗", "✅", "✨", "⭐"]
        # for entity_token in entity_tokens:
        #     self.tokenizer._additional_special_tokens.append(entity_token)

        # entity1_starts = []
        # entity2_starts = []

        # for idx in range(len(self.data)):
        #     entity1_starts.append(self.data.iloc[idx, 9])
        #     entity2_starts.append(self.data.iloc[idx, 10])

        # entities = []
        marked_sentences = []
        entity1_starts = []
        entity2_starts = []
        entity1_ends = []
        entity2_ends = []
        for idx in range(len(self.data)):
            sentence, entity1, e1s, e1e, entity2, e2s, e2e = self.data.iloc[idx, 1:8]
            if e1s < e2s:
                sentence = (
                    sentence[:e1s] + f' ❗ {entity1} ✅ '
                  + sentence[e1e + 1:] + f' ✨ {entity2} ⭐ ' + sentence[e2e: + 1:]
                )
            else:
                sentence = (
                    sentence[:e2s] + f' ✨ {entity2} ⭐ '
                  + sentence[e2e + 1:] + f' ❗ {entity1} ✅ ' + sentence[e1e + 1:]
                )
            marked_sentences.append(sentence)
            # entities.append(entity1 + '[SEP]' + entity2)

            tokenized_sentence = self.tokenizer.encode(sentence)
            entity1_token_start = tokenized_sentence.index(self.tokenizer.encode("❗")[1])
            entity2_token_start = tokenized_sentence.index(self.tokenizer.encode("✨")[1])
            entity1_token_end = tokenized_sentence.index(self.tokenizer.encode("✅")[1])
            entity2_token_end = tokenized_sentence.index(self.tokenizer.encode("⭐")[1])

            # assert entity1_token_end < self.max_length, entity1_token_end
            # assert entity2_token_end < self.max_length, entity2_token_end

            entity1_starts.append(entity1_token_start)
            entity2_starts.append(entity2_token_start)
            entity1_ends.append(entity1_token_end)
            entity2_ends.append(entity2_token_end)
            

        self.tokenized_data = self.tokenizer(
            text=marked_sentences,
            # text_pair=,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        self.tokenized_data['last_head1_indices'] = torch.tensor(entity1_starts)
        self.tokenized_data['last_head2_indices'] = torch.tensor(entity2_starts)

        return entity1_starts, entity1_ends, entity2_starts, entity2_ends




class EntityPreMarkedAndEncodedDatasetForElectra(EntityPreMarkedDatasetForElectra):
    def tokenize(self):
        entity1_starts, entity1_ends, entity2_starts, entity2_ends = super().tokenize()
        for i, (e1_start, e1_end, e2_start, e2_end) in enumerate(zip(entity1_starts, entity1_ends, entity2_starts, entity2_ends)):
            self.tokenized_data['attention_mask'][i, e1_start+1:e1_end] += 1
            self.tokenized_data['attention_mask'][i, e2_start+1:e2_end] += 1



class EntityPreMarkedDatasetForXLMRoberta(BasicDatasetForElectra):
    def tokenize(self):
        # entity_tokens = ["❗", "✅", "✨", "⭐"]
        # for entity_token in entity_tokens:
        #     self.tokenizer._additional_special_tokens.append(entity_token)

        # entity1_starts = []
        # entity2_starts = []

        # for idx in range(len(self.data)):
        #     entity1_starts.append(self.data.iloc[idx, 9])
        #     entity2_starts.append(self.data.iloc[idx, 10])

        # entities = []
        marked_sentences = []
        entity1_starts = []
        entity2_starts = []
        entity1_ends = []
        entity2_ends = []
        for idx in range(len(self.data)):
            sentence, entity1, e1s, e1e, entity2, e2s, e2e = self.data.iloc[idx, 1:8]
            if e1s < e2s:
                sentence = (
                    sentence[:e1s] + f' 😀 {entity1} ದ '
                  + sentence[e1e + 1:] + f' 😃 {entity2} ឆ្ន ' + sentence[e2e: + 1:]
                )
            else:
                sentence = (
                    sentence[:e2s] + f' 😃 {entity2} ឆ្ន '
                  + sentence[e2e + 1:] + f' 😀 {entity1} ದ ' + sentence[e1e + 1:]
                )
            marked_sentences.append(sentence)
            # entities.append(entity1 + '[SEP]' + entity2)

            tokenized_sentence = self.tokenizer.encode(sentence)
            entity1_token_start = tokenized_sentence.index(self.tokenizer.encode("😀")[1])
            entity2_token_start = tokenized_sentence.index(self.tokenizer.encode("😃")[1])
            entity1_token_end = tokenized_sentence.index(self.tokenizer.encode("ದ")[1])
            entity2_token_end = tokenized_sentence.index(self.tokenizer.encode("ឆ្ន")[1])

            assert entity1_token_end < self.max_length, entity1_token_end
            assert entity2_token_end < self.max_length, entity2_token_end

            entity1_starts.append(entity1_token_start)
            entity2_starts.append(entity2_token_start)
            entity1_ends.append(entity1_token_end)
            entity2_ends.append(entity2_token_end)
            

        self.tokenized_data = self.tokenizer(
            text=marked_sentences,
            # text_pair=,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        self.tokenized_data['last_head1_indices'] = torch.tensor(entity1_starts)
        self.tokenized_data['last_head2_indices'] = torch.tensor(entity2_starts)

        return entity1_starts, entity1_ends, entity2_starts, entity2_ends



class EntityPreMarkedAndEncodedDatasetForXLMRoberta(EntityPreMarkedDatasetForXLMRoberta):
    def tokenize(self):
        entity1_starts, entity1_ends, entity2_starts, entity2_ends = super().tokenize()
        for i, (e1_start, e1_end, e2_start, e2_end) in enumerate(zip(entity1_starts, entity1_ends, entity2_starts, entity2_ends)):
            self.tokenized_data['attention_mask'][i, e1_start:e1_end+1] += 1
            self.tokenized_data['attention_mask'][i, e2_start:e2_end+1] += 1








class ClassifiedEntityMarkedDatasetForElectra(BasicDatasetForElectra):
    def add_entity_tokens(self):
        pass

    
    def tokenize(self):
        entities = []
        marked_sentences = []
        for idx in range(len(self.data)):
            sentence, entity1, e1s, e1e, entity2, e2s, e2e, relation = self.data.iloc[idx, 1:]
            if e1s < e2s:
                sentence = sentence[:e1s] + f'[E1-{relation}]{entity1}[/E1-{relation}]' \
                     + sentence[e1e + 1:] + f'[E2-{relation}]{entity2}[/E2-{relation}]' + sentence[e2e: + 1:]
            else:
                sentence = sentence[:e2s] + f'[E2-{relation}]{entity2}[/E2-{relation}]' \
                     + sentence[e2e + 1:] + f'[E1-{relation}]{entity1}[/E1-{relation}]' + sentence[e1e + 1:]
            marked_sentences.append(sentence)
            entities.append(entity1 + '[SEP]' + entity2)

        self.tokenized_data = self.tokenizer(
            text=entities,
            text_pair=marked_sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )





# class UpscaledDataset(BasicDatasetForElectra):
#     def __init__(
#         self
#     ):
#         """Need to add parameter :labeled: to implement TTA."""
#         super(UpscaledDataset, self).__init__()
#         pass


#     def __getitem__(self, idx):
#         pass
        

#     def upscale_data_with_random_aug(self):
#         upscaled_data = []
        
#         num_augments = len(self.augments) + 1
#         num_classes = self.__class__.NUM_CLASSES

#         labels_on_data = [int(path.split('/')[-1][:2]) for path in self.data]
#         class_counter = Counter(labels_on_data)
#         max_class, max_count = class_counter.most_common(1)[0]

#         for class_idx in range(num_classes):
#             data_in_class = np.array([path for path in self.data if int(path.split('/')[-1][:2]) == class_idx])
#             random_indices = np.random.randint(0, class_counter[class_idx], max_count)
#             random_augments = np.random.randint(0, num_augments + 1, max_count)
#             upscaled_data += zip(data_in_class[random_indices], random_augments)
        
#         self.data = upscaled_data


#     def __str__(self):
#         return f"Dataset upscaled with random augmentations. {self.original_size} -> {len(self)}"
