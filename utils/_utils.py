import os
import re
import json
import shutil
import random
import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm
from tqdm import trange
from copy import deepcopy
from functools import partial
from itertools import product
from itertools import combinations
from collections import defaultdict
from hyperpyyaml import load_hyperpyyaml

from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import evaluate
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load('recall')
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")
roc_auc_metric = evaluate.load("roc_auc")
roc_auc_metric_multiclass = evaluate.load("roc_auc", "multiclass")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

