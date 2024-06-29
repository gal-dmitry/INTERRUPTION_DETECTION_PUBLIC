import sys
import os.path as op
sys.path.append(
    op.abspath(op.join(__file__, op.pardir, op.pardir))
)

from utils._utils import *
from utils.metrics import *
from utils.dataset import KFoldDataset
from utils.plotting import get_steps, plot_kfold_training 


"""
Utils
"""
def load_args(config_name):
    with open(config_name) as file:
        args = load_hyperpyyaml(file)
    return args


def get_curr_time():
    return datetime.datetime.utcnow().strftime("%Y_%m_%d__%H_%M_%S")


def create_folder(name, _dir="."):
    folder = f"{_dir}/{name}"
    os.makedirs(folder)
    return folder


def save_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
        

def get_sgd(model, weight_decay=0.01, lr=3e-5):
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": weight_decay,
        }
    ]
    
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr)
    return optimizer


def freeze_head(model):
    print("frozen head !")
    for param in model.bert.parameters():
        param.requires_grad = False
        

def freeze_except_output(model):
    print("frozen except output !")
    pattern = "encoder\.layer\.[0-9]+\.output"
    for name, param in model.bert.named_parameters():
        if not re.match(pattern, name):
            param.requires_grad = False
        

"""
Custom trainer
"""
class CustomTrainer(Trainer):
    
    
    def __init__(self, class_weights=torch.tensor([1.0, 1.0]), **kwds):
        super().__init__(**kwds)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        device = self.model.device
        inputs = inputs.to(device)
        
        labels = inputs.get("labels")
        outputs = model(**inputs)                         
        logits = outputs.get("logits")
        
        labels = labels.view(-1)
        logits = logits.view(-1, self.model.config.num_labels)
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
   

    
"""
TRAINING PROCESS
"""        
def test(test_data, model, x_column="text", x_column_pair="text_pair", batch_size=2):
    
    total_size = len(test_data)
    
    texts = []
    text_pairs = []
    y_proba = []
#     print(test_data)
#     print(test_data.__dict__)
    if "labels" in test_data.features:
        y_true = test_data["labels"]
    else:
        y_true = [np.nan for _ in range(total_size)]
        
    for i in trange(0, total_size, batch_size):

        batch = test_data[i: i + batch_size]

        text = batch[x_column]
        texts.extend(text)
        if x_column_pair:
            text_pair = batch[x_column_pair]
            text_pairs.extend(text_pair)

        input_ids = torch.tensor(batch["input_ids"]).to(model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
            proba = torch.nn.functional.softmax(logits, dim=1)
            
        # binary
        if model.num_labels == 2:
            proba = proba[:, 1] ### pos_label = 1 ?

        # multi-class
        elif model.num_labels > 2:
            proba = proba ### list in dataframe cell

        else:
            raise NotImplementedError()
                
        y_proba.extend(proba.tolist())
    
    df = pd.DataFrame({"text": texts, "y_true": y_true, "y_proba": y_proba})
    if text_pairs:
        df["text_pair"] = text_pairs
        df = df[["text", "text_pair", "y_true", "y_proba"]]
    return df


def tr_search(
    kfold_val_proba, 
    kfold_test_proba, 
    preds_dir="./", 
    num_labels=2, 
    metric="f1", 
    metric_kwargs=dict(average="binary", pos_label=1),
):

    val_tr_search, best_metric, best_tr = \
    get_best_threshold(kfold_val_proba, metric=metric, num_labels=num_labels, metric_kwargs=metric_kwargs)
    
    print("best_val_tr:", best_tr)
    print(f"best_val_{metric}:", best_metric)

    val_tr_search.to_csv(f"{preds_dir}/val_tr_search_{metric}.csv", index=False)

    kfold_val_proba["tr"] = best_tr
    kfold_test_proba["tr"] = best_tr
    
    if num_labels == 2:
        kfold_val_proba["y_pred"] = kfold_val_proba.y_proba.apply(lambda x: p2p(x, tr=best_tr))
        kfold_test_proba["y_pred"] = kfold_test_proba.y_proba.apply(lambda x: p2p(x, tr=best_tr))
    else:
        kfold_val_proba["y_pred"] = kfold_val_proba.y_proba.apply(lambda x: mp2p(x, tr=best_tr))
        kfold_test_proba["y_pred"] = kfold_test_proba.y_proba.apply(lambda x: mp2p(x, tr=best_tr))
        
    kfold_val_proba.to_csv(f"{preds_dir}/val_predictions_{metric}.csv", index=False)
    kfold_test_proba.to_csv(f"{preds_dir}/test_predictions_{metric}.csv", index=False)

    # metrics
    val_metrics = get_avg_metrics(kfold_val_proba, tr=best_tr, num_labels=num_labels)
    val_metrics.to_csv(f"{preds_dir}/val_metrics_{metric}.csv", index=False)

    test_metrics = get_avg_metrics(kfold_test_proba, tr=best_tr, num_labels=num_labels)
    test_metrics.to_csv(f"{preds_dir}/test_metrics_{metric}.csv", index=False)

    
"""
TRAIN
"""    
def train(config):
    
    # args    
    device = config["device"]
    tokenizer = config["tokenizer"]
    frozen_head = config["frozen_head"]
    frozen_except_output = False if "frozen_except_output" not in config.keys() else config["frozen_except_output"]
    
    batch_size = config["inference_batch_size"] if "inference_batch_size" in config.keys() else config["eval_batch_size"] 
    data_collator = config["data_collator"]
    training_args = config["training_args"]
    compute_metrics = config["compute_metrics"]
    preds_dir = config["preds_dir"]
    
    x_column = config["x_column"]
    x_column_pair = config["x_column_pair"] if "x_column_pair" in config.keys() else None
    
    # dataset
    df = config["df"]
    dataset = config["dataset"]
    test_fold = config["test_fold"]
    synthesize_foos = config["synthesize_foos"] 
    kfold_dataset = KFoldDataset(df, dataset, test_fold=test_fold, synthesize_foos=synthesize_foos)
    test_data = kfold_dataset.prepare_test()
    
    # cross validation
    plot_dct = {}
    kfold_val_proba = []
    kfold_test_proba = []
    
    for val_fold in sorted(config["val_folds"]):
        print(f"--------- FOLD: {val_fold} ---------")
        
        # data
        train_data, val_data = kfold_dataset.prepare_train_val(val_fold=val_fold)
        
        # model
        model = deepcopy(config["model"])
        model.to(device)        
        print("model device:", model.device)
        
        if frozen_head:
            freeze_head(model)
        elif frozen_except_output:
            freeze_except_output(model)
            
        # trainer
        output_dir = f"{config['ckpt_dir']}/{val_fold}"
        args = TrainingArguments(output_dir, **training_args)
        
        trainer_args = dict(
            class_weights=config["class_weights"],
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        if "optimizers" in config.keys():
            trainer_args["optimizers"] = config["optimizers"] 
                            
        trainer = CustomTrainer(**trainer_args)

        # train
        trainer.train()
        plot_dct[val_fold] = get_steps(trainer)

        print("--------- RESTORING WEIGHTS ---------")
        model_path = trainer.state.best_model_checkpoint
        model = model.from_pretrained(model_path)
        model.to(device)
        
        # eval
        val_df = test(val_data, model, x_column=x_column, x_column_pair=x_column_pair, batch_size=batch_size)
        val_df["fold"] = val_fold
        kfold_val_proba.append(val_df)
        
        # test
        test_df = test(test_data, model, x_column=x_column, x_column_pair=x_column_pair, batch_size=batch_size)
        test_df["fold"] = val_fold
        kfold_test_proba.append(test_df)
        
    # all folds
    save_json(plot_dct, f"{preds_dir}/plot_dct.json")
    plot_kfold_training(plot_dct, metric_names=[config["metric_name"]], suffix="", save_dir=preds_dir)
    
    kfold_val_proba = pd.concat(kfold_val_proba, axis=0)
    kfold_test_proba = pd.concat(kfold_test_proba, axis=0)
    
    # tr search
    tr_search(kfold_val_proba, 
              kfold_test_proba, 
              preds_dir=preds_dir, 
              num_labels=config["num_labels"],
              metric=config["final_metric_name"], 
              metric_kwargs=config["final_metric_kwargs"])
    

def kfold_predict(config):    
    
    folds_dct = config["folds_dct"]
        
    device = config["device"]
    x_column = config["x_column"]
    x_column_pair = config["x_column_pair"] if "x_column_pair" in config.keys() else None
    batch_size = config["batch_size"]
    
    pred_fold = config["pred_fold"]
    df = config["df"]
    df = df[df.fold == pred_fold]
    print("df.shape:", df.shape)
    
    dataset_class = config["dataset_class"]
    dataset_kwargs = config["dataset_kwargs"]
    
    kfold_proba = []
    
    for fold, model_path in folds_dct.items():
        print(f"--------- FOLD: {fold} ---------")
        
        tokenizer = deepcopy(config["tokenizer"])
        tokenizer = tokenizer.from_pretrained(model_path)
        
        model = deepcopy(config["model"])
        model = model.from_pretrained(model_path)
        model.to(device)
    
        dataset = dataset_class(tokenizer, **dataset_kwargs)
        dataset.prepare_test(df)
        test_data = dataset.test_dataset["test"]
        
        test_df = test(test_data, model, x_column=x_column, x_column_pair=x_column_pair, batch_size=batch_size)
        test_df["fold"] = fold
        kfold_proba.append(test_df)
    
    kfold_proba = pd.concat(kfold_proba, axis=0)
    kfold_proba.to_csv(config["save_path"], index=False)
    print("done !")





