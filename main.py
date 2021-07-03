import sys
sys.path.append('/opt/ml/code/stage2')

# from transformers import TrainingArguments, Trainer
from stage2.config import load_config_from_yaml
from stage2.utils import empty_logs
from stage2.train import Trainer

import argparse
# from pathlib import Path


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action='store_true')
    argparser.add_argument("--valid", action='store_true')
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--empty_logs", action='store_true')
    argparser.add_argument("--config", type=str, default="default.yaml")
    argparser.add_argument("--resuming_state", type=str, default="")

    args = argparser.parse_args()

    if args.empty_logs:
        empty_logs()
    
    config = load_config_from_yaml(args.config)
    if args.resuming_state:
        config.train.resuming_state = args.resuming_state
    
    trainer = Trainer(config)

    if args.train:
        trainer.train_and_save()
    if args.valid:
        checkpoints = []
        if checkpoints:
            for checkpoint in checkpoints:
                trainer.config.train.resuming_state = checkpoint
                trainer.load_state_dicts()
                trainer.valid()
                trainer.infer_and_save_result()
        else:
            trainer.valid()
    if args.eval:
        trainer.infer_and_save_result()
