import os
import torch
import numpy as np
from modAL.models import ActiveLearner
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from acquisition_functions import uniform, max_entropy, min_entropy, bald, var_ratios, mean_std


def active_learning_procedure(
    query_strategy,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_init: np.ndarray,
    y_init: np.ndarray,
    estimator,
    T: int = 100,
    n_query: int = 10,
    training: bool = True,
):
    """Active Learning Procedure

    Attributes:
        query_strategy: Choose between Uniform(baseline), max_entropy, bald,
        X_val, y_val: Validation dataset,
        X_test, y_test: Test dataset,
        X_pool, y_pool: Query pool set,
        X_init, y_init: Initial training set data points,
        estimator: Neural Network architecture, e.g. CNN,
        T: Number of MC dropout iterations (repeat acqusition process T times),
        n_query: Number of points to query from X_pool,
        training: If False, run test without MC Dropout (default: True)
    """
    learner = ActiveLearner(
        estimator=estimator,
        X_training=X_init,
        y_training=y_init,
        query_strategy=query_strategy,
    )
    perf_hist = [learner.score(X_test, y_test)]
    for index in range(T):
        query_idx, query_instance = learner.query(
            X_pool, n_query=n_query, T=T, training=training
        )
        learner.teach(X_pool[query_idx], y_pool[query_idx])
        X_total = np.concatenate((learner.X_training, X_pool), axis=0)
        X_total_flatten = X_total.reshape((X_total.shape[0], -1))
        y_total = np.concatenate((learner.y_training, y_pool), axis=0)
        pca = PCA(n_components=50, random_state=0)
        feature_reduced = pca.fit_transform(X_total_flatten)
        tsne = TSNE(n_components=2, random_state=0)
        feature_reduced_tsne = tsne.fit_transform(feature_reduced)

        training_pool_size = learner.y_training.shape[0]
        plt.figure(figsize=(15, 15))
        plt.scatter(feature_reduced_tsne[learner.X_training.shape[0]:, 0], feature_reduced_tsne[learner.X_training.shape[0]:, 1], c=y_pool, cmap='binary', marker='s')
        plt.scatter(feature_reduced_tsne[:learner.X_training.shape[0], 0], feature_reduced_tsne[:learner.X_training.shape[0], 1], c=learner.y_training, cmap='tab10')
        plt.title(training_pool_size)
        plt.savefig(os.path.join('tsne_images', str(training_pool_size)+'.jpg'))
        plt.show()

        X_pool = np.delete(X_pool, query_idx, axis=0)
        print(X_pool.shape)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy_val = learner.score(X_val, y_val)
        if (index + 1) % 5 == 0:
            print(f"Val Accuracy after query {index+1}: {model_accuracy_val:0.4f}")
        perf_hist.append(model_accuracy_val)
    model_accuracy_test = learner.score(X_test, y_test)
    print(f"********** Test Accuracy per experiment: {model_accuracy_test} **********")
    return perf_hist, model_accuracy_test


def select_acq_function(acq_func: int = 0) -> list:
    """Choose types of acqusition function

    Attributes:
        acq_func: 0-all(unif, max_entropy, bald), 1-unif, 2-maxentropy, 3-bald, \
                  4-var_ratios, 5-mean_std
    """
    acq_func_dict = {
        0: [uniform, max_entropy, min_entropy, bald, var_ratios, mean_std],
        1: [uniform],
        2: [max_entropy],
        3: [min_entropy],
        4: [bald],
        5: [var_ratios],
        6: [mean_std],
    }
    return acq_func_dict[acq_func]
