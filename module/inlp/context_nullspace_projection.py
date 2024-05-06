import numpy as np
import sklearn
from sklearn.svm import LinearSVC

from sklearn import cluster
from sklearn import metrics
import torch
import scipy
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from module.inlp import debias

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_gender_features(
    model,
    tokenizer,
    male_sentences,
    female_sentences,
    neutral_sentences,
):
   
    model.to(device)

    male_features = []
    female_features = []
    neutral_features = []

    # Encode the sentences.
    with torch.no_grad():
        for sentence in tqdm(male_sentences, desc="Encoding male sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            male_features.append(outputs)

        for sentence in tqdm(female_sentences, desc="Encoding female sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            female_features.append(outputs)

        for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            neutral_features.append(outputs)

    male_features = np.array(male_features)
    female_features = np.array(female_features)
    neutral_features = np.array(neutral_features)
    #print("features are:")
    #print("male features:")
    #print(male_features)
    #print("female features:")
    #print(female_features)
    #print("neutral features:")
    #print(neutral_features)

    return male_features, female_features, neutral_features


def _extract_binary_features(model, tokenizer, bias_sentences, neutral_sentences):
   
    model.to(device)

    bias_features = []
    neutral_features = []

    # Encode the sentences.
    with torch.no_grad():
        for sentence in tqdm(bias_sentences, desc="Encoding bias sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            bias_features.append(outputs)

        for sentence in tqdm(neutral_sentences, desc="Encoding neutral sentences"):
            input_ids = tokenizer(
                sentence, add_special_tokens=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**input_ids)["last_hidden_state"]
            outputs = torch.mean(outputs, dim=1)
            outputs = outputs.squeeze().detach().cpu().numpy()

            neutral_features.append(outputs)

    bias_features = np.array(bias_features)
    neutral_features = np.array(neutral_features)
    #print("bias features:")
    #print(bias_features)
    #print("neutral features:")
    #print(neutral_features)
    return bias_features, neutral_features


def _split_gender_dataset(male_feat, female_feat, neut_feat):
    np.random.seed(0)

    X = np.concatenate((male_feat, female_feat, neut_feat), axis=0)

    y_male = np.ones(male_feat.shape[0], dtype=int)
    y_female = np.zeros(female_feat.shape[0], dtype=int)
    y_neutral = -np.ones(neut_feat.shape[0], dtype=int)

    y = np.concatenate((y_male, y_female, y_neutral))

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0
    )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def _split_binary_dataset(bias_feat, neut_feat):
    np.random.seed(0)

    X = np.concatenate((bias_feat, neut_feat), axis=0)

    y_bias = np.ones(bias_feat.shape[0], dtype=int)
    y_neutral = np.zeros(neut_feat.shape[0], dtype=int)

    y = np.concatenate((y_bias, y_neutral))

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0
    )

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def _apply_nullspace_projection(
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test, n_classifiers=80
):
    classifier_parameters = {
        "fit_intercept": False,
        "class_weight": None,
        "dual": False,
        "random_state": 0,
    }

    P, rowspace_projs, Ws = debias.get_debiasing_projection(
        classifier_class=LinearSVC,
        cls_params=classifier_parameters,
        num_classifiers=n_classifiers,
        input_dim=768,
        is_autoregressive=True,
        min_accuracy=0,
        X_train=X_train,
        Y_train=Y_train,
        X_dev=X_dev,
        Y_dev=Y_dev,
        Y_train_main=None,
        Y_dev_main=None,
        by_class=False,
        dropout_rate=0,
    )

    return P, rowspace_projs, Ws

def debias_effect_analysis(P, rowspace_projs, Ws, X_train, X_dev, X_test, Y_train, Y_dev, Y_test):

    def tsne(vecs, labels, title="", ind2label=None):
        tsne = TSNE(n_components=2)
        vecs_2d = tsne.fit_transform(vecs)
        #print("labels: ",labels) 
        # Determine unique labels and corresponding names
        unique_labels = sorted(list(set(labels.tolist())))
        print("unique labels: ",unique_labels) 
        names = [ind2label[l] if ind2label else str(l) for l in unique_labels]

        plt.figure(figsize=(6, 5))
        colors = ["red", "blue", "green", "orange", "purple"]  # Add more colors if needed
        markers = ["s", "o", "^", "x", "d"]  # Add more markers if needed
        for i, label in enumerate(unique_labels):
            plt.scatter(vecs_2d[labels == label, 0], vecs_2d[labels == label, 1],
                        c=colors[i % len(colors)],
                        label=names[i],
                        alpha=0.3,
                        marker=markers[i % len(markers)])

        plt.title(title)
        plt.legend(loc="upper right")
        plt.savefig(f"embeddings_{title}.png", dpi=600)
        plt.show()
        return vecs_2d

    # Define category labels using descriptive names for religious categories
    ind2label = {
        0: "Class 0",
        1: "Class 1",
        -1: "Class 2"
    } 
    all_significantly_biased_vecs = np.concatenate((X_train, X_dev, X_test))
    all_significantly_biased_labels = np.concatenate((Y_train, Y_dev, Y_test))

    # Perform t-SNE before debiasing
    tsne_before = tsne(all_significantly_biased_vecs, all_significantly_biased_labels,
                       title="Before Debiasing", ind2label=ind2label)

    # Apply debiasing projection to data vectors
    all_significantly_biased_cleaned = P.dot(all_significantly_biased_vecs.T).T

    # Perform t-SNE after debiasing
    tsne_after = tsne(all_significantly_biased_cleaned, all_significantly_biased_labels,
                      title="After Debiasing", ind2label=ind2label)

    def perform_purity_test(vecs, k, labels_true):
        np.random.seed(0)
        clustering = sklearn.cluster.KMeans(n_clusters=k,n_init=10)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        score = sklearn.metrics.homogeneity_score(labels_true, labels_pred)
        return score

    def compute_v_measure(vecs, labels_true, k=2):
        np.random.seed(0)
        clustering = sklearn.cluster.KMeans(n_clusters=k)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        return sklearn.metrics.v_measure_score(labels_true, labels_pred)

    # remove neutral class, keep only male and female biased
    X_dev = X_dev[Y_dev != -1]
    X_train = X_train[Y_train != -1]
    X_test = X_test[Y_test != -1]

    Y_dev = Y_dev[Y_dev != -1]
    Y_train = Y_train[Y_train != -1]
    Y_test = Y_test[Y_test != -1]

    X_dev_cleaned = (P.dot(X_dev.T)).T
    X_test_cleaned = (P.dot(X_test.T)).T
    X_trained_cleaned = (P.dot(X_train.T)).T

    print("V-measure-before (TSNE space): {}".format(compute_v_measure(tsne_before, all_significantly_biased_labels)))
    print("V-measure-after (TSNE space): {}".format(compute_v_measure(tsne_after, all_significantly_biased_labels)))

    print("V-measure-before (original space): {}".format(
        compute_v_measure(all_significantly_biased_vecs, all_significantly_biased_labels), k=2))
    print("V-measure-after (original space): {}".format(compute_v_measure(X_test_cleaned, Y_test), k=2))

    rank_before = np.linalg.matrix_rank(X_train)
    rank_after = np.linalg.matrix_rank(X_trained_cleaned)
    print("Rank before: {}; Rank after: {}".format(rank_before, rank_after))

def compute_projection_matrix(model, tokenizer, data, bias_type, n_classifiers=80):
   
    if bias_type == "gender":
        male_sentences = data["male"]
        female_sentences = data["female"]
        neutral_sentences = data["neutral"]

        male_features, female_features, neutral_features = _extract_gender_features(
            model, tokenizer, male_sentences, female_sentences, neutral_sentences
        )

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = _split_gender_dataset(
            male_features, female_features, neutral_features
        )

    else:
        bias_sentences = data["bias"]
        neutral_sentences = data["neutral"]

        bias_features, neutral_features = _extract_binary_features(
            model, tokenizer, bias_sentences, neutral_sentences
        )

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = _split_binary_dataset(
            bias_features, neutral_features
        )

    print("Dataset split sizes:")
    print(
        f"Train size: {X_train.shape[0]}; Dev size: {X_dev.shape[0]}; Test size: {X_test.shape[0]}"
    )

    P, rowspace_projs, Ws = _apply_nullspace_projection(
        X_train, X_dev, X_test, Y_train, Y_dev, Y_test, n_classifiers=n_classifiers
    )
    debias_effect_analysis(P, rowspace_projs, Ws, X_train, X_dev, X_test, Y_train, Y_dev, Y_test)
    P = torch.tensor(P, dtype=torch.float32)

    return P
