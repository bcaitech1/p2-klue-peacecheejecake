import random
import numpy as np
import os
import torch


SEED = 77

random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_see_dall(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
