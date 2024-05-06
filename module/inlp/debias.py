import random
import sys
from typing import Dict
from typing import List
import warnings

import numpy as np
import scipy
from tqdm import tqdm

from module.inlp import classifier


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # Orthogonal basis

    P_W = w_basis.dot(w_basis.T)  # Orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(
    rowspace_projection_matrices: List[np.ndarray], input_dim: int
):
   
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)

    return P


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    
    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


def get_debiasing_projection(
    classifier_class,
    cls_params: Dict,
    num_classifiers: int,
    input_dim: int,
    is_autoregressive: bool,
    min_accuracy: float,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
    by_class=False,
    Y_train_main=None,
    Y_dev_main=None,
    dropout_rate=0,
) -> np.ndarray:
 
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn(
            "Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds."
        )

    I = np.eye(input_dim)

    if by_class:
        if (Y_train_main is None) or (Y_dev_main is None):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers))
    for i in pbar:

        clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1.0 / (1 - dropout_rate + 1e-6)
        dropout_mask = (np.random.rand(*X_train.shape) < (1 - dropout_rate)).astype(
            float
        ) * dropout_scale

        if by_class:
            # cls = np.random.choice(Y_train_main)  # uncomment for frequency-based sampling
            cls = random.choice(main_task_labels)
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network(
            (X_train_cp * dropout_mask)[relevant_idx_train],
            Y_train[relevant_idx_train],
            X_dev_cp[relevant_idx_dev],
            Y_dev[relevant_idx_dev],
        )
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy:
            continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W)  # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:

            P = get_projection_to_intersection_of_nullspaces(
                rowspace_projections, input_dim
            )
            # project

            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

   P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws


if __name__ == "__main__":

    from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression

    N = 10000
    d = 300
    X = np.random.rand(N, d) - 0.5
    Y = np.array(
        [1 if sum(x) > 0 else 0 for x in X]
    ) 
    num_classifiers = 200
    classifier_class = SGDClassifier  
    input_dim = d
    is_autoregressive = True
    min_accuracy = 0.0

    P, rowspace_projections, Ws = get_debiasing_projection(
        classifier_class,
        {},
        num_classifiers,
        input_dim,
        is_autoregressive,
        min_accuracy,
        X,
        Y,
        X,
        Y,
        by_class=False,
    )

    I = np.eye(P.shape[0])
    P_alternative = I - np.sum(rowspace_projections, axis=0)
    P_by_product = I.copy()

    for P_Rwi in rowspace_projections:

        P_Nwi = I - P_Rwi
        P_by_product = P_Nwi.dot(P_by_product)

    """testing"""

    
    if is_autoregressive:
        assert np.allclose(P_alternative, P)
        assert np.allclose(P_by_product, P)

    
    assert np.allclose(P.dot(P), P)

    
    x = np.random.rand(d) - 0.5
    for w in Ws:

        assert np.allclose(np.linalg.norm(w.dot(P.dot(x))), 0.0)

   
    if is_autoregressive:
        for i, w in enumerate(Ws):

            for j, w2 in enumerate(Ws):

                if i == j:
                    continue

                assert np.allclose(np.linalg.norm(w.dot(w2.T)), 0)
