import pandas as pd 
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#create_folds

df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

df.loc[:,"kfold"] = -1

## random shuffle 
df = df.sample(frac=1).reset_index(drop=True)

targets = df.drop("sig_id", axis=1).values

NFOLDS = 8
mskf = MultilabelStratifiedKFold(n_splits = NFOLDS)
for fold, (trn, val) in enumerate(mskf.split(X=df, y = targets)):
    if fold != 7:
        df.loc[val, "kfold"] = fold
    else:
        df.loc[val, "kfold"] = "hold"
    
df.to_csv("../folds/train_targets_folds.csv", index=False)
print("made ", NFOLDS, " folds")