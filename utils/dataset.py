import sys
import os.path as op
sys.path.append(
    op.abspath(op.join(__file__, op.pardir, op.pardir))
)

from utils._utils import *



"""
Synthesis
"""
def _split(texts):
    texts_cutted = []
    for text in texts:
        texts_cutted += [t.strip() for t in re.split(',|!|\.|\?', text)]
    texts_cutted = [t for t in texts_cutted if t != ""]
    texts_cutted = list(set(texts_cutted) - set(texts))
    return texts_cutted


def _merge(texts, merge_elements=3):
    texts_merged = []
    for i in trange(2, merge_elements+1):
        pairs = combinations(texts, i)
        pairs = [" ".join(pair) for pair in pairs]
        texts_merged += pairs
    texts_merged = list(set(texts_merged) - set(texts))
        
    return texts_merged
         
        
def _crop(texts, crop_offset=3):
    cropped_texts = set()
    texts = set(texts)
    for text in tqdm(texts):
        text_len = len(text.split())
        for start in range(0, crop_offset+1):
            for end in range(0, crop_offset+1):
                if start + end == 0:
                    continue
                if start + end >= text_len:
                    continue
                cropped_text = " ".join(text.split()[start:-end])
                cropped_texts.add(cropped_text)
    cropped_texts = list(cropped_texts - texts)
    return cropped_texts
    
    
def _synthesize(texts, split=True, merge_elements=3, crop_offset=3):
    
    new_texts = deepcopy(texts)
    
    # 1. split on punctuation marks
    if split:
        print("split...")
        texts_cutted = _split(new_texts)
        new_texts += texts_cutted
    
    # 2. merge
    if merge_elements:
        print("merge...")
        texts_merged = _merge(new_texts, merge_elements=merge_elements)
        new_texts += texts_merged
    
    # 3. crop
    if crop_offset:
        print("crop...")
        texts_cropped = _crop(new_texts, crop_offset=crop_offset)
        new_texts += texts_cropped
    
    # 4. remove duplicates and intersections
    new_texts = list(set(new_texts) - set(texts))
    
    return new_texts


def synthesize_text(df, 
                      text_col="text",
                      label_col="y_true",
                      label="Backchannels", 
                      fold="synthetic",
                      synthesize_kwargs=dict(
                          split=True, 
                          merge_elements=3, 
                          crop_offset=3
                          ),
                      random_state=42,
                      no_more_than_opposite=False,
                      max_multiplier=False):
    
    
    neg_df = df[df[label_col] != label]
    pos_df = df[df[label_col] == label]
    diff = neg_df.shape[0] - pos_df.shape[0]
    if no_more_than_opposite and diff <= 0:
        print(f"the num of {label} is greater than the num of opposite class")
        return df
    
    texts = pos_df[text_col].tolist()
    new_texts = _synthesize(texts, **synthesize_kwargs)
    
    upsampled_pos_df = pd.DataFrame({text_col: new_texts, label_col: label, "fold": fold})
    if max_multiplier:
        additional = pos_df.shape[0] * (max_multiplier - 1)
        additional = min(additional, upsampled_pos_df.shape[0])
        upsampled_pos_df = upsampled_pos_df.sample(additional, random_state=random_state)
        
    if no_more_than_opposite and upsampled_pos_df.shape[0] > diff:
        upsampled_pos_df = upsampled_pos_df.sample(diff, random_state=random_state)
        
    df = pd.concat([neg_df, pos_df, upsampled_pos_df], axis=0)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def synthesize_pair(df, 
                    text_col="text",
                    text_pair_col="text_pair",
                    label_col="y_true",
                    label="Backchannels", 
                    fold="synthetic",
                    relation="elementwise",
                    no_more_than_opposite=True,
                    synthesize_text_kwargs=dict(
                      split=True, 
                      merge_elements=False, 
                      crop_offset=3
                      ),
                    synthesize_text_pair_kwargs=dict(
                      split=True, 
                      merge_elements=3, 
                      crop_offset=3
                      ),
                    random_state=42,
                    max_multiplier=False):
    
    # separate
    neg_df = df[df[label_col] != label]
    pos_df = df[df[label_col] == label]
    diff = neg_df.shape[0] - pos_df.shape[0]
    if no_more_than_opposite and diff <= 0:
        print(f"the num of {label} is greater than the num of opposite class")
        return df
    
    # synthesize
    if relation == "product":
        texts = pos_df[text_col].tolist()
        text_pairs = pos_df[text_pair_col].tolist()
        
        new_texts = _synthesize(texts, **synthesize_text_kwargs)
        new_text_pairs = _synthesize(text_pairs, **synthesize_text_pair_kwargs)
        
        new_texts += texts
        new_text_pairs += text_pairs
        
        pairs = product(new_texts, new_text_pairs)
        new_texts, new_text_pairs = zip(*pairs)
    
    elif relation == "elementwise":
        
        new_texts = []
        new_text_pairs = []
        
        for _, row in pos_df.iterrows():
            texts = [row[text_col]]
            text_pairs = [row[text_pair_col]]
            
            new_texts_ = _synthesize(texts, **synthesize_text_kwargs)
            new_text_pairs_ = _synthesize(text_pairs, **synthesize_text_pair_kwargs)
            
            new_texts_ += texts
            new_text_pairs_ += text_pairs
        
            inner_diff = len(new_texts_) - len(new_text_pairs_)
            if inner_diff > 0:
                new_text_pairs_ += random.choices(new_text_pairs_, k=inner_diff)
            elif inner_diff < 0:
                new_texts_ += random.choices(new_texts_, k=-inner_diff)

            new_texts.extend(new_texts_)
            new_text_pairs.extend(new_text_pairs_)
    
    else:
        raise NotImplementedError()
        
    # concat
    upsampled_pos_df = pd.DataFrame({text_col: new_texts, text_pair_col: new_text_pairs, label_col: label, "fold": fold})
    
    if max_multiplier:
        additional = pos_df.shape[0] * (max_multiplier - 1)
        additional = min(additional, upsampled_pos_df.shape[0])
        upsampled_pos_df = upsampled_pos_df.sample(additional, random_state=random_state)
        
    if no_more_than_opposite and upsampled_pos_df.shape[0] > diff:
        upsampled_pos_df = upsampled_pos_df.sample(diff, random_state=random_state)
        
    df = pd.concat([neg_df, pos_df, upsampled_pos_df], axis=0)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df



"""
Utils
"""
def get_train_test(df, test_fold=0):
    train_df = df[df.fold != test_fold]
    test_df = df[df.fold == test_fold]
    return train_df, test_df


def remove_intersection(train_df, test_df, cols=["text"]):
    intersection = pd.merge(train_df[cols], test_df[cols], on=cols, how="inner")
    index = (train_df.isin(intersection).sum(axis=1) < len(cols))
    train_df = train_df[index]
    return train_df



"""
DATASET
"""
class AbstractDataset:
    
    def prepare_test(self, test_df):
        dataset = DatasetDict()
        dataset["test"] = self._df2dataset(test_df)
        dataset = dataset.map(self._tokenize, batched=True)
        self.test_dataset = dataset
    
    def prepare_train_val(self, train_df, val_df):
        dataset = DatasetDict()
        dataset["train"] = self._df2dataset(train_df)
        dataset["val"] = self._df2dataset(val_df)
        dataset = dataset.map(self._tokenize, batched=True)
        self.train_val_dataset = dataset
    

class ImprovedAbstractDataset:
    
    def prepare_test(self, test_df):
        dataset = DatasetDict()
        dataset["test"] = self._df2dataset(test_df)
        dataset = dataset.map(self._tokenize, batched=False)
        self.test_dataset = dataset
    
    def prepare_train_val(self, train_df, val_df):
        dataset = DatasetDict()
        dataset["train"] = self._df2dataset(train_df)
        dataset["val"] = self._df2dataset(val_df)
        dataset = dataset.map(self._tokenize, batched=False)
        self.train_val_dataset = dataset
        
        
class BertDataset(AbstractDataset):
    
    def __init__(self, 
                 tokenizer,
                 tokenizer_args=dict(max_length=512, truncation=True, padding=True),
                 x_column="text", 
                 y_column="label"):
        
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.x_column=x_column
        self.y_column=y_column    
        
        
    def _df2dataset(self, df):
        df = df.reset_index(drop=True)
        df = df[[self.x_column, self.y_column]]
        df = df.rename(columns={self.y_column: "labels"})
        dataset = Dataset.from_pandas(df)
        return dataset
    
    
    def _tokenize(self, dataset):
        return self.tokenizer(dataset[self.x_column], **self.tokenizer_args)
        

class PairBertDataset(AbstractDataset):
    
    def __init__(self, 
                 tokenizer,
                 tokenizer_args=dict(max_length=512, truncation=True, padding=True),
                 x_column="text_prev",
                 x_column_pair="text", 
                 y_column="label"):
        
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.x_column = x_column
        self.x_column_pair = x_column_pair
        self.y_column = y_column   
        
        
    def _df2dataset(self, df):
        df = df.reset_index(drop=True)
        df = df[[self.x_column, self.x_column_pair, self.y_column]]
        df = df.rename(columns={self.y_column: "labels"})
        dataset = Dataset.from_pandas(df)
        return dataset
    
    
    def _tokenize(self, dataset):
        return self.tokenizer(dataset[self.x_column], dataset[self.x_column_pair], **self.tokenizer_args)
        

class ImprovedPairBertDataset(ImprovedAbstractDataset):
# class ImprovedPairBertDataset(AbstractDataset):
    
    def __init__(self, 
                 tokenizer,
                 symmetric_padding_side=False,
                 tokenizer_args=dict(max_length=512, truncation=True, padding=True),
                 x_column="text_prev",
                 x_column_pair="text", 
                 y_column="label"):
        
        self.symmetric_padding_side = symmetric_padding_side
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.x_column = x_column
        self.x_column_pair = x_column_pair
        self.y_column = y_column   
        
        
    def _df2dataset(self, df):
        df = df.reset_index(drop=True)
        df = df[[self.x_column, self.x_column_pair, self.y_column]]
        df = df.rename(columns={self.y_column: "labels"})
        dataset = Dataset.from_pandas(df)
        return dataset
    
    
    def _tokenize(self, dataset):
        tokenizer_args = deepcopy(self.tokenizer_args)
        tokenizer_args["max_length"] = tokenizer_args["max_length"] // 2
        
        print("max_length", tokenizer_args["max_length"])
        
        self.tokenizer.truncation_side = "left"
        if self.symmetric_padding_side:
            self.tokenizer.padding_side = "left"
        left = self.tokenizer(dataset[self.x_column], **tokenizer_args)
        
        self.tokenizer.truncation_side = "right"
        if self.symmetric_padding_side:
            self.tokenizer.padding_side = "right"
        right = self.tokenizer(dataset[self.x_column_pair], **tokenizer_args)
        
        result = dict()
        result["input_ids"] = left["input_ids"] + right["input_ids"]
        result["token_type_ids"] = [0 for _ in left["token_type_ids"]] + [1 for _ in right["token_type_ids"]]
        result["attention_mask"] = left["attention_mask"] + right["attention_mask"]
        
        print("input_ids", len(result["input_ids"]))
        print("token_type_ids", len(result["token_type_ids"]))
        print("attention_mask", len(result["attention_mask"]))
        
        return result
        
        
        
"""
Inference
"""
class InferenceBertDataset(AbstractDataset):
    
    def __init__(self, 
                 tokenizer,
                 tokenizer_args=dict(max_length=512, truncation=True, padding=True),
                 x_column="text"):
        
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.x_column=x_column

        
    def _df2dataset(self, df):
        df = df.reset_index(drop=True)
        df = df[[self.x_column]]
        dataset = Dataset.from_pandas(df)
        return dataset
    
    
    def _tokenize(self, dataset):
        return self.tokenizer(dataset[self.x_column], **self.tokenizer_args)
    
    
    
"""
KFOLD DATASET
"""    
class KFoldDataset:
    
    def __init__(self, df, dataset, test_fold=0, synthesize_foos=[]):
        for synthesize_foo in synthesize_foos:
            print("synthesize foo:", type(synthesize_foo), synthesize_foo)
        self.synthesize_foos = synthesize_foos
        self.dataset = dataset
        self.df, self.test_df = get_train_test(df, test_fold=test_fold)
    
    def prepare_test(self):
        self.dataset.prepare_test(self.test_df)
        test_data = self.dataset.test_dataset["test"]
        return test_data
        
    def prepare_train_val(self, val_fold=0):
        train_df, val_df = get_train_test(self.df, test_fold=val_fold)
        for synthesize_foo in self.synthesize_foos:
            print(f"train size before synthesis: {train_df.shape[0]}")
            train_df = synthesize_foo(train_df)
            print(f"train size after synthesis: {train_df.shape[0]}")
            cols = []
            if hasattr(self.dataset, "x_column"):
                cols.append(self.dataset.x_column)
            if hasattr(self.dataset, "x_column_pair"):
                cols.append(self.dataset.x_column_pair)
            train_df = remove_intersection(train_df, val_df, cols=cols)
            train_df = remove_intersection(train_df, self.test_df, cols=cols)
            print(f"train size after intersection removing: {train_df.shape[0]}", end="\n\n")
            print(train_df[self.dataset.y_column].value_counts())
            
        self.dataset.prepare_train_val(train_df, val_df)
        train_data = self.dataset.train_val_dataset["train"]
        val_data = self.dataset.train_val_dataset["val"]
        return train_data, val_data
    
    
    
        