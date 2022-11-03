DATA_FOLDS = "predictions"
DATA_RESULTS = "resico_dl"

def write_results(path, results):
    with open(path, "w") as f:
        for result in results:
            f.writelines(result + "\n")

for index in range(10):
    results_1 = []
    results_3 = []
    results_5 = []

    with open("{}/fold_{}.csv".format(DATA_FOLDS, index), "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line = line.strip()
                line_divided = line.split(",")
                y_true = int(float(line_divided[0]))

                values_str = line_divided[1:]
                values = []
                values.append(int(values_str[0][2:]))
                values.append(int(values_str[1].strip()))
                values.append(int(values_str[2].strip()))
                values.append(int(values_str[3].strip()))
                values.append(int(values_str[4].strip()[:-2]))

                # Evaluation for Top-1
                if y_true == values[0]:
                    results_1.append(str(y_true) + "," + str(y_true))
                else:
                    results_1.append(str(y_true) + "," + str(values[0]))

                # Evaluation for Top-3
                if y_true in values[:3]:
                    results_3.append(str(y_true) + "," + str(y_true))
                else:
                    results_3.append(str(y_true) + "," + str(values[0]))

                # Evaluation for Top-5
                if y_true in values:
                    results_5.append(str(y_true) + "," + str(y_true))
                else:
                    results_5.append(str(y_true) + "," + str(values[0]))
    
    write_results("{}/{}/fold_{}.txt".format(DATA_RESULTS, 1, index), results_1)
    write_results("{}/{}/fold_{}.txt".format(DATA_RESULTS, 3, index), results_3)
    write_results("{}/{}/fold_{}.txt".format(DATA_RESULTS, 5, index), results_5)
