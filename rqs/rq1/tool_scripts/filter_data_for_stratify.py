import pandas as pd

from sys import argv

if __name__ == '__main__':
    if len(argv) < 2:
        print("Missing the parameter of the data file")
    elif len(argv) == 2:
        data_file_path = argv[1]
        filename = data_file_path.split(".")[0]

        print("Reading the data file ...")
        data_pd = pd.read_csv(data_file_path)
        print("Done!")

        print("Getting the value counts of the class column ...")
        value_counts = data_pd[data_pd.columns[-1]].value_counts()
        print("Done!")

        print("Getting the classes with a higher or equal number of repetitions ...")
        higher_equal_10 = list(value_counts[value_counts >= 10].index)
        print("Done!")

        print("Getting the data from the indices ...")
        data_stratified = data_pd[data_pd[data_pd.columns[-1]].isin(higher_equal_10)]
        print("Done!")

        print("Saving the stratified data ...")
        data_stratified.to_csv(filename + "_strat.csv", index=False)
        print("Done!")
