import torch
import random
import numpy as np
from copy import deepcopy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from datasets import DatasetDict
from datasets.utils.logging import disable_progress_bar

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
disable_progress_bar()


"""
Dataset
"""
class InferenceBertDataset:
    def __init__(
        self,
        tokenizer,
        tokenizer_args=dict(max_length=128, truncation=True, padding=True),
        x_column="text",
    ):
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.x_column = x_column

    def _df2dataset(self, df):
        df = df[[self.x_column]]
        dataset = Dataset.from_pandas(df)
        return dataset

    def _tokenize(self, dataset):
        return self.tokenizer(dataset[self.x_column], **self.tokenizer_args)

    def prepare_data(self, df):
        dataset = DatasetDict()
        dataset["test"] = self._df2dataset(df)
        dataset = dataset.map(self._tokenize, batched=True)
        self.dataset = dataset

        

"""
Proba 2 Pred
"""
def p2p(y, tr=0.5):
    return 1 if y >= tr else 0
    
def proba2pred(y_proba, tr=0.5):
    y_pred = [p2p(y, tr=tr) for y in y_proba]
    return y_pred

def mp2p(y, tr=0.5):
    return 1 + np.argmax(y[1:]) if np.sum(y[1:]) >= tr else 0

def multiclass_proba2pred(y_proba, tr=0.5):
    y_pred = [mp2p(y, tr=tr) for y in y_proba]
    return y_pred



"""
Model
"""
class InferenceBert:
    def __init__(self, args, device="cuda:0"):
        self.args = args
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args['model_path'])
        assert self.model.num_labels >= 2
        self.model.to(device)
        self.model.eval()
        self.dataset = InferenceBertDataset(
            tokenizer=AutoTokenizer.from_pretrained(self.args['model_path']),
            tokenizer_args=self.args['tokenizer_args'],
            x_column=self.args['x_column'],
        )
        self.tr = self.args["tr"]
        self.batch_size = self.args["batch_size"]
        self.label2id = self.args["label2id"]
        self.return_proba = self.args["return_proba"]
        
        
    def predict(self, df):
        df = df.copy()
        dataset = deepcopy(self.dataset)
        dataset.prepare_data(df)
        
        batch_size = self.batch_size
        total_size = dataset.dataset["test"].shape[0]
        
        proba = []
        for i in range(0, total_size, batch_size):
            batch = dataset.dataset["test"][i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(self.model.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(self.model.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
                probas = torch.nn.functional.softmax(logits, dim=1)
            proba.append(probas)
                
        proba = torch.vstack(proba)
        proba = proba.numpy()
        
        if self.model.num_labels == 2:
            proba = proba[:, 1]
            pred = proba2pred(proba, tr=self.tr)

        else:
            pred = multiclass_proba2pred(proba, tr=self.tr)
        
        if self.return_proba:
            proba = [p for p in proba]
            df["y_proba"] = proba
        
        labels = [self.label2id[label] for label in pred]
        df["y_pred"] = labels

        return df

    def __call__(self, df):
        return self.predict(df)
