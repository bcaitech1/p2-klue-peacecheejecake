from stage2.utils.log import CSVLogger
from .branch import ConfigBranch

import os
import yaml
import torch

try:
    Loader = yaml.CLoader
except:
    Loader = yaml.Loader


class YAMLConfigParser:
    @staticmethod
    def load(config_path):
        with open(config_path) as yamlfile:
            config = yaml.load(yamlfile, Loader=Loader)

        config = ConfigBranch(**config)
        return config

    @staticmethod
    def load_separate_all(system, path, data, train, model):
        with open(system) as yamlfile:
            system = yaml.load(yamlfile, Loader=Loader)
        with open(path) as yamlfile:
            path = yaml.load(yamlfile, Loader=Loader)
        with open(data) as yamlfile:
            data = yaml.load(yamlfile, Loader=Loader)
        with open(train) as yamlfile:
            train = yaml.load(yamlfile, Loader=Loader)
        with open(model) as yamlfile:
            model = yaml.load(yamlfile, Loader=Loader)
        
        config = ConfigBranch()

        config.system = ConfigBranch(**system)
        config.path = ConfigBranch(**path)
        config.data = ConfigBranch(**data)
        config.train = ConfigBranch(**train)
        config.model = ConfigBranch(**model)

        return config
    
    @staticmethod
    def save(config, saving_path):
        with open(saving_path, 'w') as yamlfile:
            yaml.dump(config, yamlfile)


def load_config_from_yaml(config_file):
    path = os.path.join('/opt/ml/code/stage2/configs', config_file)
    config = YAMLConfigParser.load(path)
    if config.train.logger == 'manual':
        config.train.logger = CSVLogger()
    return config
