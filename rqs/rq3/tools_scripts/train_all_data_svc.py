import time
import pickle
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import softmax

class LinearSVCwithProba(LinearSVC):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)

if __name__ == "__main__":
    DIM = 20
    MODELS_FOLDER = "models"

    print("Loading the data ...")
    data = pd.read_csv("data.csv", header=None).to_numpy()

    X = data[:, :DIM]
    y = data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
    print("Done!")

    classifier_conf = LinearSVCwithProba(C=11.13591703457844, dual=False, max_iter=1e5)
    classifier = OneVsRestClassifier(classifier_conf, n_jobs=-1)

    print("Training ...")
    start_time = time.time()
    classifier.fit(X, y)
    end_time = time.time()

    print("Training time for the fold {} seconds.".format(end_time - start_time))
    print("Done!")

    print("Saving the model for future extrinsic evaluations ...")
    pickle.dump(classifier, open("{}/svc.pickle".format(MODELS_FOLDER), "wb"))
    print("Done!")
