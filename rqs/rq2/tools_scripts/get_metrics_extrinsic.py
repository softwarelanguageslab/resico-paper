import numpy as np
from sys import argv
from sklearn.metrics import f1_score, precision_score, recall_score


def get_predictions(path: str):
    y = list()
    y_hat = list()

    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                line_divided = line.split(",")

                y.append(line_divided[0])
                y_hat.append(line_divided[1])

    return y, y_hat


if __name__ == '__main__':
    if len(argv) < 2:
        print("Missing dataset name :(")
    elif len(argv) == 2:
        RESULTS_FOLDER = "/Users/kmilo/Dev/PhD/RESICO_new/rq2/results/coster"
        DATASET_NAME = argv[1]

        top_ks = [1, 3, 5]

        for top_k in top_ks:
            print("Analysing results for top-{0} of extrinsic dataset {1} ...".format(top_k, DATASET_NAME))
            print()
            RESULTS_TOP = "{0}/{1}".format(RESULTS_FOLDER, top_k)
            PATH_DATASET = "{0}/{1}.txt".format(RESULTS_TOP, DATASET_NAME)

            y_true, y_pred = get_predictions(PATH_DATASET)

            precision = precision_score(y_true, y_pred, average="micro", labels=np.unique(y_pred))
            recall = recall_score(y_true, y_pred, average="micro", labels=np.unique(y_pred))
            f1 = f1_score(y_true, y_pred, average="micro", labels=np.unique(y_pred))

            print("Precision (micro): %.2f" % precision)
            print("Recall (micro): %.2f" % recall)
            print("F1-Score (micro): %.2f" % f1)
            print()
