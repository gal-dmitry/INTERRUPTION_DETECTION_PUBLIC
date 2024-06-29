import sys
import os.path as op
sys.path.append(
    op.abspath(op.join(__file__, op.pardir, op.pardir))
)

from utils._utils import *
from utils.metrics import _get_metrics, _get_avg_metrics


def get_ensemble_predictions_with_false(df1, df2, false_replace=1):
    
    folds = sorted(df1.fold.unique())
    
    dfs = []
    for fold in folds:
        _df1 = df1[df1.fold == fold]
        _df2 = df2[df2.fold == fold]
        
        _df = pd.merge(_df1, _df2, on=["text"], how="left", suffixes=('_1', '_2'))
        _df["y_true"] = np.nan
        _df.loc[_df.y_true_1 == 0, "y_true"] = 0
        _df.loc[_df.y_true_2 == 0, "y_true"] = 1
        _df.loc[_df.y_true_2 == 1, "y_true"] = 2
        _df.y_true = _df.y_true.astype(int)
        
        _df["y_pred"] = np.nan
        _df.loc[_df.y_pred_1 == 0, "y_pred"] = 0
        _df.loc[(_df.y_pred_1 == 1) & (_df.y_pred_2 == 0), "y_pred"] = 1
        _df.loc[(_df.y_pred_1 == 1) & (_df.y_pred_2 == 1), "y_pred"] = 2
        _df.loc[_df.y_pred.isna(), "y_pred"] = false_replace
        _df.y_pred = _df.y_pred.astype(int)
        _df["fold"] = _df.fold_1.copy()
        
        dfs.append(_df)
    
    result = pd.concat(dfs, axis=0)
    return result


def get_avg_ensemble_metrics(df):
        
    kfold_dct = {}
    folds = sorted(df.fold.unique())
    for fold in folds:
        _df = df[df.fold == fold]
        y_true = _df.y_true.tolist() 
        y_pred = _df.y_pred.tolist()
        kfold_dct[fold] = _get_metrics(y_true, y_pred, binary=False)

    result = _get_avg_metrics(kfold_dct)
    return result
    
    
    