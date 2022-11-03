
import pickle
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import optimizers

from pyspark import SparkContext, SparkConf
from elephas.ml_model import ElephasEstimator
from elephas.ml.adapter import to_data_frame

from pyspark.ml import Pipeline
from pyspark.sql.functions import *

DIM = 20
batch_size = 10000
nb_classes = 4860
epochs = 200
MODELS_FOLDER = "models"

adam = optimizers.Adam(lr=0.01)
adam_conf = optimizers.serialize(adam)

print("Configuring the Spark Context ...")
conf = SparkConf().setAppName('RESICO').set("spark.driver.memory", "20g").setMaster('local[16]')
sc = SparkContext(conf=conf)
print("Done!")

print("Loading the data ...")
data = pd.read_csv("data.csv", header=None).to_numpy()

X = data[:, :DIM]
y = data[:, DIM] - 1 # The classes are lowered 1 since the minimum in the data is 1 and not zero, this is done for evaluation purposes

# Changing the types of the target
y = y.astype('uint64')

# Converting to categorical data
y = to_categorical(y, nb_classes)
print("Done!")

model = Sequential()
model.add(Dense(nb_classes, input_dim=20))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Configuring the data ...")
df = to_data_frame(sc, X, y, categorical=True)

# Initialize Spark ML Estimator
estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_json())
estimator.set_optimizer_config(adam_conf)
estimator.set_mode("synchronous")
estimator.set_loss("categorical_crossentropy")
estimator.set_metrics(['acc'])
estimator.set_epochs(epochs)
estimator.set_batch_size(batch_size)
estimator.set_validation_split(0.1)
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)
estimator.set_verbosity(1)

# Fitting a model returns a Transformer
pipeline = Pipeline(stages=[estimator])
print("Training ...")
fitted_pipeline = pipeline.fit(df)
print("Done!")

print("Saving the model for future extrinsic evaluations ...")
pickle.dump(fitted_pipeline, open("{}/dl.pickle".format(MODELS_FOLDER), "wb"))
print("Done!")
