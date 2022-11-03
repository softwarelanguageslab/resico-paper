import time
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    DIM = 20
    MODELS_FOLDER = "models"

    print("Loading the data ...")
    data = pd.read_csv("data.csv", header=None).to_numpy()

    X = data[:, :DIM]
    y = data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
    print("Done!")

    classifier = KNeighborsClassifier(
        n_neighbors=2,
        weights="distance",
        algorithm="kd_tree",
        leaf_size=63
    )

    print("Training ...")
    start_time = time.time()
    classifier.fit(X, y)
    end_time = time.time()

    print("Training time: {} seconds.".format(end_time - start_time))
    print("Done!")

    print("Saving the model for future extrinsic evaluations ...")
    pickle.dump(classifier, open("{}/knn.pickle".format(MODELS_FOLDER), "wb"))
    print("Done!")
