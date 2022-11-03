import optuna

import pandas as pd
from sklearn.metrics import f1_score

from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

print("Loading the data ...")

dim = 20

dataset = pd.read_csv("data.csv", header=None).to_numpy()
X = dataset[:, :dim]
y = dataset[:, dim]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=42)
print("Done!")
    
def objective_linear(trial):
    linear_alpha = trial.suggest_float("linear_alpha", 1.0, 1e10, log=True)
    linear_solver = trial.suggest_categorical("linear_solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])

    classifier_obj = RidgeClassifier(alpha=linear_alpha, solver=linear_solver)

    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)
    f1_score_metric = f1_score(y_test, y_pred, average="micro")

    return 1 - f1_score_metric

def objective_ovr_svc(trial):
    svc_C = trial.suggest_float("svc_C", 1.0, 1e10, log=True)

    classifier = LinearSVC(C=svc_C, dual=False, max_iter=1e5)
    classifier_obj = OneVsRestClassifier(classifier)

    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)
    f1_score_metric = f1_score(y_test, y_pred, average="micro")

    return 1 - f1_score_metric

def objective_knn(trial):
    knn_neighbours = trial.suggest_int("knn_neighbours", 2, 1e3, log=True)
    knn_weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    knn_algorithm = trial.suggest_categorical("knn_algorithm", ["ball_tree", "kd_tree", "brute"])
    knn_leaf_size = trial.suggest_int("knn_leaf", 2, 1e3, log=True)

    classifier_obj = KNeighborsClassifier(
        n_neighbors=knn_neighbours,
        weights=knn_weights,
        algorithm=knn_algorithm,
        leaf_size=knn_leaf_size
    )

    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)
    f1_score_metric = f1_score(y_test, y_pred, average="micro")

    return 1 - f1_score_metric

study = optuna.create_study(direction="minimize")
study.optimize(objective_knn, n_trials=200, gc_after_trial=True, n_jobs=-1)
study.optimize(objective_linear, n_trials=200, gc_after_trial=True, n_jobs=-1)
study.optimize(objective_ovr_svc, n_trials=200, gc_after_trial=True, n_jobs=-1)

print(study.best_trial)
print(study.best_value)
trials_df = study.trials_dataframe()
trials_df.to_csv("hpo_knn.csv", index=False)
