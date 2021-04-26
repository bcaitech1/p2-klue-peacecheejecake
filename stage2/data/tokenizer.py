import sys
import transformers


class AutoTokenizer:
    @staticmethod
    def from_pretrained(class_name, model_id):
        try:
            PreTrainedTokenizer = getattr(sys.modules[__name__], class_name)
            ## TODO: customize tokenizers
            raise NotImplementedError
        except:
            PreTrainedTokenizer = getattr(transformers, class_name)
            tokenizer = PreTrainedTokenizer.from_pretrained(model_id)
        
        return tokenizer


def add_special_tokens(tokenizer, model, add_type: str = 'simple'):
    if add_type == 'simple':
        tokens_to_add = {"additional_special_tokens": ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    elif add_type == 'ner':
        raise NotImplementedError
    tokenizer.add_special_tokens(tokens_to_add)
    model.resize_token_embeddings(
        # tokenizer.vocab_size + tokenizer.add_special_tokens(tokens_to_add)
        len(tokenizer)
    )
    