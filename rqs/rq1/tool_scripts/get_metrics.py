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

                y.append(int(line_divided[0]))
                y_hat.append(int(line_divided[1]))

    return y, y_hat


if __name__ == '__main__':
    if len(argv) < 2:
        print("Missing model results name :(")
    elif len(argv) == 2:
        RESULTS_FOLDER = "results"
        MODEL_NAME = argv[1]
        PATH_RESULTS = RESULTS_FOLDER + "/" + MODEL_NAME

        top_ks = [1, 3, 5]

        for top_k in top_ks:
            print("Analysing results for top {0} ...".format(top_k))
            print()
            RESULTS_TOP = "{0}/{1}".format(PATH_RESULTS, top_k)

            precisions = list()
            recalls = list()
            f1_scores = list()

            folds = list(range(1, 11))
            for fold in folds:
                # print("Results for fold {0} ...".format(fold))
                FOLD_PATH = "{0}/fold_{1}.txt".format(RESULTS_TOP, fold)

                y_true, y_pred = get_predictions(FOLD_PATH)

                precision_fold = precision_score(y_true, y_pred, average="micro", labels=np.unique(y_pred))
                recall_fold = recall_score(y_true, y_pred, average="micro", labels=np.unique(y_pred))
                f1_fold = f1_score(y_true, y_pred, average="micro", labels=np.unique(y_pred))

                # print("Precision (micro): %.2f" % precision_fold)
                # print("Recall (micro): %.2f" % recall_fold)
                # print("F1-Score (micro): %.2f" % f1_fold)

                precisions.append(precision_fold)
                recalls.append(recall_fold)
                f1_scores.append(f1_fold)

            average_precision = sum(precisions) / len(precisions)
            average_recall = sum(recalls) / len(recalls)
            average_f1 = sum(f1_scores) / len(f1_scores)

            print("Average Precision (micro): %.2f" % average_precision)
            print("Average Recall (micro): %.2f" % average_recall)
            print("Average F1-Score (micro): %.2f" % average_f1)
            print()
