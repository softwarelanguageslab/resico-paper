import time
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier

class RidgeClassifierwithProba(RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        probs = np.exp(d) / np.sum(np.exp(d))
        return probs

if __name__ == "__main__":
    DIM = 20
    MODELS_FOLDER = "models"

    print("Loading the data ...")
    data = pd.read_csv("data.csv", header=None).to_numpy()

    X = data[:, :DIM]
    y = data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
    print("Done!")

    classifier = RidgeClassifierwithProba(
        alpha=6473.182223729128,
        solver="sag"
    )

    print("Training ...")
    start_time = time.time()
    classifier.fit(X, y)
    end_time = time.time()

    print("Training time for the fold {} seconds.".format(end_time - start_time))
    print("Done!")

    print("Saving the model for future extrinsic evaluations ...")
    pickle.dump(classifier, open("{}/linear.pickle".format(MODELS_FOLDER), "wb"))
    print("Done!")
