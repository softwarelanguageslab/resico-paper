import time
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

class RidgeClassifierwithProba(RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        probs = np.exp(d) / np.sum(np.exp(d))
        return probs

if __name__ == "__main__":
    FOLDER_STRAT = "strat"
    DIM = 20

    print("Intrinsic Training, evaluating and saving the results of the sampled and stratified dataset ...", flush=True)
    for index in range(10):
        print("Fold {} / 10 ...".format(index + 1), flush=True)

        print("Loading the data ...", flush=True)
        train_data_path = "{0}/train_{1}.csv".format(FOLDER_STRAT, index)
        test_data_path = "{0}/test_{1}.csv".format(FOLDER_STRAT, index)

        train_data = pd.read_csv(train_data_path, header=None).to_numpy()
        test_data = pd.read_csv(test_data_path, header=None).to_numpy()

        X_train = train_data[:, :DIM]
        y_train = train_data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes

        X_test = test_data[:, :DIM]
        y_test = test_data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
        print("Done!", flush=True)

        classifier = RidgeClassifierwithProba(
            alpha=6473.182223729128,
            solver="sag"
        )

        print("Training ...", flush=True)
        start_time = time.time()
        classifier.fit(X_train, y_train)
        end_time = time.time()

        print("Training time for the fold {} seconds.".format(end_time - start_time))
        print("Done!", flush=True)

        top_ks = (1, 3, 5)

        for top_k in top_ks:
            print("Evaluating the top {}".format(top_k), flush=True)
 
            with open("results/intrinsic/linear/{}/fold_{}.txt".format(top_k, index + 1), "w") as f:
                print("Getting the prediction matrix ...", flush=True)
                probs_matrix = classifier.predict_proba(X_test)
                print("Done!", flush=True)

                for i, probs in enumerate(probs_matrix):
                    if i % 6000 == 0:
                        print("{} / {}".format(i, probs_matrix.shape[0]), flush=True)

                    probs_arr = [round(float(value), 2) for value in list(probs)]
                    max_index = probs_arr.index(max(probs_arr))
                    true_value = int(y_test[i])

                    indexes = list(np.argpartition(probs, -top_k)[-top_k:])

                    if true_value in indexes:
                        f.write(str(true_value) + "," + str(true_value) + "\n")
                    else:
                        f.write(str(true_value) + "," + str(max_index) + "\n")
    print("Done!")
