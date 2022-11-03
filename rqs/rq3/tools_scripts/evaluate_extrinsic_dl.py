import pickle
import pandas as pd
from keras.utils import to_categorical
from elephas.ml.adapter import to_data_frame
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import *

def processing_rows(row):
    best_values = sorted(row.prediction, reverse=True)[:5]
    indexes_values = [row.prediction.index(value) for value in best_values]

    return row.label, indexes_values

if __name__ == "__main__":
    model = "dl"
    dataset_names = ("COSTER-SO", "StatType-SO", "RESICO-SO")

    DATA_FOLDER_TRANS = "datasets-resico-trans"
    MODELS_FOLDER = "models"
    nb_classes = 4860
    DIM = 20

    print("Loading the model ...")
    classifier = pickle.load(open("{}/{}.pickle".format(MODELS_FOLDER, model), "rb"))
    print("Done!")

    print("Configuring the Spark Context ...")
    conf = SparkConf().setAppName('RESICO').set("spark.driver.memory", "20g").setMaster('local[16]')
    sc = SparkContext(conf=conf)
    print("Done!")

    for dataset_name in dataset_names:
        dataset = "{}-T.csv".format(dataset_name)

        print("Loading the data ...")
        data = pd.read_csv("{}/{}".format(DATA_FOLDER_TRANS, dataset), header=None).to_numpy()

        X_test = data[:, :DIM]
        y_test = data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes
        y_test = y_test.astype('uint64')
        y_test = to_categorical(y_test, nb_classes)
        print("Done!")

        print("Configuring the test data ...")
        test_df = to_data_frame(sc, X_test, y_test, categorical=True)
        print("Done!")

        print("Loading the saved classifier ...")
        fitted_pipeline = pickle.load(open("{}/dl.pickle".format(MODELS_FOLDER), "rb"))
        print("Done!")

        prediction = fitted_pipeline.transform(test_df)
        pnl = prediction.select("label", "prediction")

        pnl_max_5_values = pnl.rdd.map(lambda row: (row.label, row.prediction.index(sorted(row.prediction, reverse=True)[:5])))

        pnl_max_5_values = pnl.rdd.map(processing_rows)
        prenl = pnl_max_5_values.toDF()
        prenl.withColumn("_2", col("_2").cast("string")).write.csv("predictions/extrinsic/{}".format(dataset_name))

        print("Done!")
