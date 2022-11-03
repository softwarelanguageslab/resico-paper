import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    FOLDER_STRAT = "strat"
    DIM = 20

    print("Intrinsic Training, evaluating and saving the results of the sampled and stratified dataset ...")
    for index in range(10):
        print("Fold {} / 10 ...".format(index + 1))

        print("Loading the data ...")
        train_data_path = "{0}/train_{1}.csv".format(FOLDER_STRAT, index)
        test_data_path = "{0}/test_{1}.csv".format(FOLDER_STRAT, index)

        train_data = pd.read_csv(train_data_path, header=None).to_numpy()
        test_data = pd.read_csv(test_data_path, header=None).to_numpy()

        X_train = train_data[:, :DIM]
        y_train = train_data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes

        X_test = test_data[:, :DIM]
        y_test = test_data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
        print("Done!")

        classifier = KNeighborsClassifier(
            n_neighbors=2,
            weights="distance",
            algorithm="kd_tree",
            leaf_size=63
        )

        print("Training ...")
        start_time = time.time()
        classifier.fit(X_train, y_train)
        end_time = time.time()

        print("Training time for the fold {} seconds.".format(end_time - start_time))
        print("Done!")

        top_ks = (1, 3, 5)

        for top_k in top_ks:
            print("Evaluating the top {}".format(top_k))

            with open("results/intrinsic/knn/{}/fold_{}.txt".format(top_k, index + 1), "w") as f:
                for i, x_test in enumerate(X_test):
                    if i % 6000 == 0:
                        print("{} / {}".format(i, X_test.shape[0]))

                    probs = classifier.predict_proba(np.array([x_test]))
                    probs_arr = [round(float(value), 2) for value in list(probs[0])]
                    
                    max_index = probs_arr.index(max(probs_arr))
                    true_value = int(y_test[i])

                    indexes = list(np.argpartition(probs[0], -top_k)[-top_k:])

                    if true_value in indexes:
                        f.write(str(true_value) + "," + str(true_value) + "\n")
                    else:
                        f.write(str(true_value) + "," + str(max_index) + "\n")

    print("Done!")
