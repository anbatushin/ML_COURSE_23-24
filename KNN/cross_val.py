import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    folds = []
    fold_len = num_objects // num_folds
    objects = np.arange(num_objects)
    for i in range(num_folds - 1):
        train_objects = np.concatenate((objects[:fold_len * i], objects[fold_len * (i + 1):]))
        test_objects = objects[fold_len * i:fold_len * (i + 1)]
        folds.append((train_objects, test_objects))
    train_objects = objects[:fold_len * (num_folds - 1)]
    test_objects = objects[fold_len * (num_folds - 1):]
    folds.append((train_objects, test_objects))

    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """

    res_dict = {}
    for n_neighbors in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weight in parameters['weights']:
                for normalizer in parameters['normalizers']:
                    res = []
                    for train, test in folds:
                        if normalizer[0]:
                            normalizer[0].fit(X[train])
                            X_scaled = normalizer[0].transform(X[train])
                            X_test_scaled = normalizer[0].transform(X[test])
                        else:
                            X_scaled = X[train]
                            X_test_scaled = X[test]

                        X_train, X_test = X_scaled, X_test_scaled
                        Y_train, Y_test = y[train], y[test]

                        clf = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                        clf.fit(X_train, Y_train)

                        Y_pred = clf.predict(X_test)

                        score = score_function(Y_test, Y_pred)
                        res.append(score)

                    res_dict[(normalizer[1], n_neighbors, metric, weight)] = np.mean(res)

    return res_dict
