from tkinter import E
import pandas as pd
from sys import argv
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    if len(argv) < 2:
        print("Not enough arguments, the name of the result file should be passed as argument.")
    elif len(argv) == 2:
        path_data = argv[1]
        name_data = path_data.split("/")[-1]
        simple_name_data = name_data.split(".")[0]

        PROCESSED_RESULTS = "/Users/kmilo/Dev/PhD/RESICO_new/rq0/results/processed"
        
        data = pd.read_csv(path_data)
        data_selected = dict()
        column_transfomations = ""
        
        data_selected['Index'] = data['number'].to_list()

        column_names = data.columns.to_list()
        param_columns = list(filter(lambda name: name.startswith('params'), column_names))

        for column in param_columns:
            divided_column_name = column.split("_")
            column_name = "_".join(divided_column_name[2:])
            column_name = column_name[0].upper() + column_name[1:]
            data_column = data[column]
            data_column_list = data_column.to_list()

            if data_column.dtype == "object":
                le = LabelEncoder()
                data_column_list = le.fit_transform(data_column_list)

                attributes_transformed = le.classes_
                trans_mapping = ["{} -> {}".format(index, attribute) for (index, attribute) in enumerate(attributes_transformed)]
                column_transfomations += "{}: {}".format(column_name, ", ".join(trans_mapping)) + "\n"

            data_selected[column_name] = data_column_list
        
        data_selected["Loss"] = data['value'].to_list()

        # Write the processed results
        df = pd.DataFrame(data=data_selected)
        df.to_csv("{}/{}".format(PROCESSED_RESULTS, name_data), index=False)

        # Write the transformations if any
        if len(column_transfomations):
            with open("{}/trans_{}.txt".format(PROCESSED_RESULTS, simple_name_data), "w") as f:
                f.writelines(column_transfomations)