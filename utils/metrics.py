import sys
import os.path as op
sys.path.append(
    op.abspath(op.join(__file__, op.pardir, op.pardir))
)


from utils._utils import *


"""
ROC AUC: https://huggingface.co/spaces/evaluate-metric/roc_auc
"""
def compute_binary_roc_auc(eval_pred):
    prediction_scores, labels = eval_pred
    prediction_scores = torch.nn.functional.softmax(torch.from_numpy(prediction_scores), dim=1).numpy()
    prediction_scores = prediction_scores[:, 1]
    results = roc_auc_metric.compute(references=labels, prediction_scores=prediction_scores)
    return results


def compute_multiclass_roc_auc(eval_pred, metric_kwargs=dict(average="macro", multi_class="ovr")):
    prediction_scores, labels = eval_pred
    prediction_scores = torch.nn.functional.softmax(torch.from_numpy(prediction_scores), dim=1).numpy()
    results = roc_auc_metric_multiclass.compute(references=labels, prediction_scores=prediction_scores, **metric_kwargs)
    return results


"""
F1: https://huggingface.co/spaces/evaluate-metric/f1
"""
def compute_f1(eval_pred, metric_kwargs=dict(average="binary", pos_label=1)):
    prediction_scores, labels = eval_pred
    predictions = np.argmax(prediction_scores, axis=1)
    return f1_metric.compute(predictions=predictions, references=labels, **metric_kwargs)


"""
Get metric function
"""
def get_metric_foo(
    metric_name="roc_auc", 
    num_labels=2, 
    metric_kwargs=dict(
        average="macro", 
        multi_class="ovr"
    ),
):
    """
    https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    """
    if metric_name == "f1":
        foo = partial(compute_f1, metric_kwargs=metric_kwargs)
 
    elif metric_name == "roc_auc":
        if num_labels == 2:
            foo = compute_binary_roc_auc
        elif num_labels > 2:
            foo = partial(compute_multiclass_roc_auc, metric_kwargs=metric_kwargs)
        else:
            raise ValueError()
    else:
        raise ValueError()
        
    return foo
            
    
    
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
BEST THRESHOLD
"""
def get_best_threshold(df, metric="f1", num_labels=2, metric_kwargs=dict(average="binary", pos_label=1)):

    if num_labels == 2:
        _proba2pred = proba2pred
        trs = sorted(df.y_proba.unique())
    else:
        _proba2pred = multiclass_proba2pred
        trs = sorted(df.y_proba.apply(lambda x: sum(x[1:])).unique())
    
    folds = sorted(df.fold.unique())
    result = pd.DataFrame(index=[trs], columns=folds)
    
    for fold in tqdm(folds, desc="search best threshold"):
        _df = df[df.fold == fold]
        y_true = _df.y_true.tolist()
        y_proba = _df.y_proba.tolist()
        for tr in tqdm(trs):
            y_pred = _proba2pred(y_proba, tr=tr)
            if metric == "f1":
                metric_value = f1_metric.compute(references=y_true, predictions=y_pred, **metric_kwargs)['f1']
            elif metric == "balanced_accuracy":
                metric_value = balanced_accuracy_score(y_true, y_pred, **metric_kwargs)
            else:
                raise ValueError()
            result.loc[tr, fold] = metric_value
            
    result["mean"] = result.apply(lambda row: np.mean([row[col] for col in folds]), axis=1)
    result["std"] = result.apply(lambda row: np.std([row[col] for col in folds]), axis=1)
    result["var"] = result.apply(lambda row: np.var([row[col] for col in folds]), axis=1)
    
    result = result.rename(columns={col: f"f1_{col}" for col in result.columns})
    result = result.reset_index()
    result = result.rename(columns={"level_0": "tr"})
    
    best_metric = result[f"{metric}_mean"].max()
    best_tr = result.loc[result[f"{metric}_mean"] == best_metric, "tr"].tolist()[0]
    
    return result, best_metric, best_tr


"""
--- REPORTS ---
"""
def _get_roc_auc(y_true, y_proba, binary=True):
    dct = dict()
    if binary:
        dct["roc_auc_binary"] = roc_auc_metric.compute(references=y_true, prediction_scores=y_proba)['roc_auc']
    else:
        dct["roc_auc_macro_ovr"] = \
        roc_auc_metric_multiclass.compute(references=y_true, prediction_scores=y_proba, average="macro", multi_class="ovr")['roc_auc']
        dct["roc_auc_macro_ovo"] = \
        roc_auc_metric_multiclass.compute(references=y_true, prediction_scores=y_proba, average="macro", multi_class="ovo")['roc_auc']
        dct["roc_auc_weighted_ovr"] = \
        roc_auc_metric_multiclass.compute(references=y_true, prediction_scores=y_proba, average="weighted", multi_class="ovr")['roc_auc']
        dct["roc_auc_weighted_ovo"] = \
        roc_auc_metric_multiclass.compute(references=y_true, prediction_scores=y_proba, average="weighted", multi_class="ovo")['roc_auc']
    return dct


def _get_metrics(y_true, y_pred, binary=True):
    dct = dict()
    if binary:
        # binary 0
        dct['Recall_0'] = recall_metric.compute(references=y_true, predictions=y_pred, average="binary", pos_label=0)['recall']
        dct['Precision_0'] = precision_metric.compute(references=y_true, predictions=y_pred, average="binary", pos_label=0)['precision']
        dct['F1_binary_0'] = f1_metric.compute(references=y_true, predictions=y_pred, average="binary", pos_label=0)['f1']
        # binary 1
        dct['Recall_1'] = recall_metric.compute(references=y_true, predictions=y_pred, average="binary", pos_label=1)['recall']
        dct['Precision_1'] = precision_metric.compute(references=y_true, predictions=y_pred, average="binary", pos_label=1)['precision']
        dct['F1_binary_1'] = f1_metric.compute(references=y_true, predictions=y_pred, average="binary", pos_label=1)['f1']
    else:
        recall = recall_metric.compute(references=y_true, predictions=y_pred, average=None)['recall']
        precision = precision_metric.compute(references=y_true, predictions=y_pred, average=None)['precision']
        f1 = f1_metric.compute(references=y_true, predictions=y_pred, average=None)['f1']
        for i, (r, p, f) in enumerate(zip(recall, precision, f1)):
            dct[f"Recall_{i}"] = r
            dct[f"Precision_{i}"] = p
            dct[f"F1_{i}"] = f
            
    # avg macro
    dct['Recall_macro'] = recall_metric.compute(references=y_true, predictions=y_pred, average="macro")['recall']
    dct['Precision_macro'] = precision_metric.compute(references=y_true, predictions=y_pred, average="macro")['precision']
    dct['F1_macro'] = f1_metric.compute(references=y_true, predictions=y_pred, average="macro")['f1']
    # avg weighted
    dct['Recall_weighted'] = recall_metric.compute(references=y_true, predictions=y_pred, average="weighted")['recall']
    dct['Precision_weighted'] = precision_metric.compute(references=y_true, predictions=y_pred, average="weighted")['precision']
    dct['F1_weighted'] = f1_metric.compute(references=y_true, predictions=y_pred, average="weighted")['f1']
    # accuracy
    dct['Balanced_Accuracy_Score'] = balanced_accuracy_score(y_true, y_pred)
    dct['Accuracy_Score'] = accuracy_metric.compute(references=y_true, predictions=y_pred)['accuracy']
    return dct
    
    
def get_binary_metrics(y_true, y_proba, tr=0.5):
    y_pred = proba2pred(y_proba, tr=tr)
    roc_auc_dct = _get_roc_auc(y_true, y_proba, binary=True)
    dct = _get_metrics(y_true, y_pred, binary=True)
    dct = dct | roc_auc_dct
    dct["tr"] = tr
    return dct


def get_multiclass_metrics(y_true, y_proba, tr=0.5):
    y_pred = multiclass_proba2pred(y_proba, tr=tr)
    roc_auc_dct = _get_roc_auc(y_true, y_proba, binary=False)
    dct = _get_metrics(y_true, y_pred, binary=False)
    dct = dct | roc_auc_dct
    dct["tr"] = tr
    return dct



"""
Averaging
"""
def _get_avg_metrics(kfold_dct, digits=4):
    
    res_dct = defaultdict(list)
    for fold, dct in kfold_dct.items():
        res_dct["fold"].append(fold)
        for metric_name, metric_value in dct.items():
            res_dct[metric_name].append(metric_value)
            
    df = pd.DataFrame(res_dct).T
    df.columns = df.iloc[0].astype(int)
    df.columns.name = None
    df = df[1:]

    mean = df.apply(lambda x: round(x.mean(), digits), axis=1)
    std = df.apply(lambda x: round(x.std(), digits), axis=1)
    var = df.apply(lambda x: round(x.var(), digits), axis=1)
    df["mean"] = mean
    df["std"] = std
    df["var"] = var
    df = df.reset_index()
    df = df.rename(columns={"index": "metric"})
    return df


def get_avg_metrics(df, tr=0.5, num_labels=2):
    
    if num_labels == 2:
        get_metrics = get_binary_metrics
    else:
        get_metrics = get_multiclass_metrics
        
    kfold_dct = {}
    folds = sorted(df.fold.unique())
    for fold in folds:
        _df = df[df.fold == fold]
        y_true = _df.y_true.tolist() 
        y_proba = _df.y_proba.tolist()
        kfold_dct[fold] = get_metrics(y_true, y_proba, tr=tr)

    result = _get_avg_metrics(kfold_dct)
    return result
    
    
    

"""
Visual Reports
"""
def _plot_matrix(matrix, labels):
    
    sns.heatmap(matrix, 
                square=True,
                cmap="YlGnBu",
                linewidths=4,
                annot=True,
                fmt=".0f",
                xticklabels=labels, 
                yticklabels=labels)
    
    plt.title("confusion_matrix")
    plt.show()

    
def get_classification_report(y_true, y_pred, plot=False):
    
    print(classification_report(y_true, y_pred, digits=4))
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    setdiff = np.setdiff1d(pred_labels, true_labels)
    if len(setdiff) > 0:
        true_labels = np.concatenate((setdiff, true_labels))
    if plot:
        matrix = confusion_matrix(y_true, y_pred)
        _plot_matrix(matrix, true_labels)

