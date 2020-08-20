import numpy as np 
import pandas as pd

from functools import partial 

from sklearn import ensemble, metrics, model_selection 

from skopt import gp_minimize, space
from skopt.plots import plot_convergence

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope 

# To find the best parameters for the best accuracy we can use algorithms that can optimize a function 
# Minimization of this function to minimize the loss 
# If you want to find the best parameters to maximize the loss you can minimizie it by multiplying it by -1 
# This way, we are minimizing the negative of accuracy but in fact we are maximizing the accuracy 

def optimize(params, param_names, x, y):
    """
    :params : list of params from gp_minimize
    :param_names : list of param names. ORDER IS IMPORTANT!
    :x : training data
    :y : labels/target
    :return : negative accuracy after 5 folds 
    """

    params = dict(zip(param_names, params))

    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []

    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]
        
        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_accuracy = metrics.accuracy_score(
                ytest,
                preds
            )
        accuracies.append(fold_accuracy)

    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv(file)

    # Price range is the targe variable 
    X= df.drop("price_range", axis = 1).values
    y = df.price_range.values

    # define a param space 

    param_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
            "n_estimators": scope.int(
                hp.quniform("n_estimators", 100, 1500, 1)
                ),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_features": hp.quniform("max_features", 0, 1)

            }
    # make a list of param names
    # SAME ORDER AS THE PARAM SPACE 

    optimization_function = partial(
            optimize, 
            x=X,
            y=y
            )

    trials = Trials()

    #run hyperopt

    hopt = fmin(
            fn=optimization_function, 
            space=param_space,
            algo=type.suggest,
            max_evals=15,
            trials=trials
            )
    print(hopt)

    
