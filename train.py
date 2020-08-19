import argparse 
import os 
import joblib 
import pandas as pd 
from sklearn import metrics
import model_dispatcher
from sklearn import model_selection

# Kfold function takes in a dataframe and converts it into a stratified k fold version 

def kfold_(df):
    df = pd.read_csv(file)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y= df.target.values
    kf= model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f


# Here the code is used to train a model on a dataframe that has already been split into relative kfolds to the relative number of target variable classes

def run(fold, model):
    df = pd.read_csv(file)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid= df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("label", axis = 1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop("label", axis = 1).values
    y_valid = df_valid.label.values

    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)
    
    preds - clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f'Fold = {fold}, Accuracy = {accuracy}')

    


