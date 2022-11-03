import numpy as np
import pandas as pd
import pickle

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.extmath import softmax


class LinearSVCwithProba(LinearSVC):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)


class RidgeClassifierwithProba(RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        probs = np.exp(d) / np.sum(np.exp(d))
        return probs

if __name__ == "__main__":
    model = "linear"
    dataset_names = ("COSTER-SO", "StatType-SO", "RESICO-SO")

    DATA_FOLDER_TRANS = "datasets-resico-trans"
    MODELS_FOLDER = "models"
    DIM = 20

    print("Loading the model ...")
    classifier = pickle.load(open("{}/{}.pickle".format(MODELS_FOLDER, model), "rb"))
    print("Done!")

    for dataset_name in dataset_names:
        dataset = "{}-T.csv".format(dataset_name)

        print("Loading the data ...")
        data = pd.read_csv("{}/{}".format(DATA_FOLDER_TRANS, dataset), header=None).to_numpy()

        X_test = data[:, :DIM]
        y_test = data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
        print("Done!")

        top_ks = (1, 3, 5)

        for top_k in top_ks:
            print("Evaluating the top {}".format(top_k))

            with open("results/extrinsic/{}/{}/{}.txt".format(model, top_k, dataset_name), "w") as f:
                print("Getting the prediction matrix ...")
                probs_matrix = classifier.predict_proba(X_test)
                print("Done!")

                for i, probs in enumerate(probs_matrix):
                    print("{} / {}".format(i + 1, probs_matrix.shape[0]))

                    probs_arr = [round(float(value), 2) for value in list(probs)]
                    max_index = probs_arr.index(max(probs_arr))
                    true_value = int(y_test[i])

                    indexes = list(np.argpartition(probs, -top_k)[-top_k:])

                    if true_value in indexes:
                        f.write(str(true_value) + "," + str(true_value) + "\n")
                    else:
                        f.write(str(true_value) + "," + str(max_index) + "\n")
        print("Done!")
