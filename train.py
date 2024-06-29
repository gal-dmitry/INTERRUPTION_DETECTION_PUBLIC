import sys
import os.path as op
sys.path.append(
    op.abspath(op.join(__file__, op.pardir, op.pardir))
)

import argparse
from utils._utils import *
from utils.training import train, load_args

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

import random
import torch
import numpy as np
import pandas as pd
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()


# args
config = load_args(args.config_path)
res_path = f"{config['main_dir']}/{args.config_path.split('/')[-1]}"
shutil.copy(args.config_path, res_path)


# train
train(config)


