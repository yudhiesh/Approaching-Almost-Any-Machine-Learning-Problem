import numpy as np 
import pandas as pd

from functools import partial 

from sklearn import ensemble, metrics, model_selection 

from skopt import gp_minimize, space
from skopt.plots import plot_convergence

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

    param_space = [
            # max_depth is an integer between 3 and 10
            space.Integer(3, 15, name="max_depth"),
            # n_estimators is an integer between 50 and 1500
            spcae.Integer(100, 1500, name="n_estimators")
            #criterion is a category
            space.Categorical(["gini", "entropy"], name ="criterion")
            # You can also have a real numbered space and define a distribution you want to pick it from 
            space.Real(0.01, 1, prior="uniform", name = "max_features")

    ]
    # make a list of param names
    # SAME ORDER AS THE PARAM SPACE 

    param_names = [
            "max_depth",
            "n_estimators]",
            "criterion",
            "max_features"
        ]
    optimization_function = partial(
            optimize, 
            param_names=param_names,
            x=X,
            y=y
            )

    # Call gp_minimize which uses a Bayesian optimization for the minimization of the optimization function 

    result = gp_minimize(
            optimization_function,
            dimensions=param_space,
            n_calls= 15,
            n_random_start =10,
            verbose = 10
            )
    # create the best params dict and print it out 
    best_params = dict(
            zip(
                param_names,
                result.x
                )
            )
    print(best_params)

# Plot results 
# plot_convergence(results)


    
