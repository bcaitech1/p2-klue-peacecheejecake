# Stage 02 - Relation Extraction (KLUE)

### Example
* 학습 후 추론하기
```bash
python main.py --train --eval --config my_config.yaml
```

## Overview
문장 내 subject와 object 사이의 관계 추출을 위한 코드입니다.

## Stage2 Package Structure

```text
stage2/
├── __init__.py
├── config/                   
│   ├── __init__.py
│   ├── parser.py             # parse yaml file to config object
│   └── tree.py               # struct config
│
├── data/                     
│   ├── __init__.py           
│   ├── tokenizer.py          # augmentations (most are unused)
│   ├── dataset.py            # different datasets for different models
│   └── functional.py         # helper functions for data processing
│
├── train/
│   ├── __init__.py
│   ├── lr_scheduler.py       # custom learning rate scheduler
│   └── trainer.py            # trainer for training, validation, 
│                             # and creating submission file
├── modules/
│   ├── __init__.py
│   ├── models/               # define or load models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── electra.py
│   │   ├── xlmr.py
│   │   └── classifier.py
│   └── loss.py               # customized loss functions
│                             
└── utils/
    ├── __init__.py
    ├── log.py                # functionalities to log on csv and plot
    ├── seed.py               # fix random seeds
    └── utils.py              # utilities

configs/                      # Some configuration files (yaml)
```

## How to Use

### Install Requiements

```shell
pip install -r requirements.txt
```

### Configurations

학습 시 사용하는 config 파일은 `yaml`파일로 학습 목표에 따라 다음과 같이 설정해주세요.

```yaml
system: &default_system_config
  device: "cuda:0"
  num_workers: 8


path: &default_path_config
  input: '/opt/ml/input'
  train: '/opt/ml/input/data/train/train.csv'
  test: '/opt/ml/input/data/test/test.csv'
  valid: 
  submit: '/opt/ml/output/submissions'
  checkpoint: '/opt/ml/output/checkpoints'


data: &default_data_config
  num_classes: 42
  dataset_class: EntityPreMarkedAndEncodedDatasetForXLMRoberta
  tokenizer_class: XLMRobertaTokenizer
  max_token_length: 365 #
  valid_ratio: 0
  upscaling: False
  sampler: 
  batch_size: 16
  num_folds: 5


train: &default_train_config
  experiment_name: xlmroberta
  resuming_state: 
    
  lr:
    base: 5.0e-6
    min: 
    scheduler: CosineAnnealingAfterWarmUpAndHardRestartScheduler
    warmup_steps: 600
    cycle_steps: 300
    damping_ratio: 0.1
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]
  momentum: 0.9
  nesterov: False
  criterion: FocalAndCrossEntropyLoss
  optimizer:
    name: AdamW
  
  num_epochs: 20
  valid_iter_limit: 0
  
  valid_min_epoch: 1
  test_min_epoch: 1
  save_min_epoch: 1

  valid_period: 1
  test_period: 0
  save_period: 1

  valid_min_acc: 0
  test_min_acc: 1
  save_min_acc: 0

  stop_count: 5

  #wandb options
  logger: neptune


model: &default_model_config
  arc: XLMRobertaForPreMarkedSequenceConcatClassification
  pretrained_id: xlm-roberta-base
  name: xlm-roberta-large-concat-clf
  last_head_idx: 0

teacher:
  arc:
  state_path: 

```

<br/>

### Train

```shell
python main.py --train [--config] [--resuming_state] [--empty_logs]
```

- `--config`: config 파일 경로를 나타냅니다. 생략할 경우 `config/default.yaml`가 지정됩니다.
- `--resuming_state`: 모델 checkpoint 경로입니다. 명시되는 경우에 이어서 학습이 진행됩니다.
- `--empty_log`: csv logging을 사용할 경우, log를 초기화합니다.

### Inference

```shell
python main.py --eval --resuming_state [--config]
```

### Validation
지정된 validation dataset에 대해 validation을 별도로 실행할 수 있습니다. 
config에 따라 별도의 validation file을 불러오거나, train dataset을 split합니다.

```shell
python main.py --valid --resuming_state [--config] [--empty_logs]
```
