import mlflow
import pandas as pd

from hyperopt import hp, tpe, fmin, STATUS_OK, SparkTrials
from hyperopt.fmin import space_eval
from hyperopt.pyll.base import scope

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("Configuring the enviornment ...")
mlflow.sklearn.autolog(disable=True)

print("Done!")

print("Loading the data ...")
dim = 20

dataset = pd.read_csv("data.csv", header=None).to_numpy()
X = dataset[:, :dim]
y = dataset[:, dim]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=42)
print("Done!")

print("Define the hyperparameter search space")
space_rf = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 10, 200, 1)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
    "min_samples_split": hp.uniform("min_samples_split", 0, 1),
}

n_evals = 200
print("Done!")

# Defining the objective function
def objective(hyperparameters):
  # Instantiate the model with hyperparameters
  model = RandomForestClassifier(**hyperparameters)

  # Train the model
  model.fit(X_train, y_train)

  # Evaluate the learned model
  val_pred = model.predict(X_test)
  val_f1_score = f1_score(y_test, val_pred, average="micro")

  # Use negative F1 score as loss metric
  return {"loss": 1 - val_f1_score, "status": STATUS_OK}

trials = SparkTrials(parallelism=32)
best = fmin(
  fn=objective,
  space=space_rf,
  algo=tpe.suggest,
  max_evals=n_evals,
  trials=trials
)

def unpack_values(trial):
    vals = trial["misc"]["vals"]
    rval = {}
    for k, v in list(vals.items()):
        if v:
            rval[k] = v[0]
    return rval

def export_data(trials):
  # filter out trials with status non-ok
  valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]

  # get the losses of the filtered trials
  losses = [float(trial['result']['loss']) for trial in valid_trial_list]

  indexes = []
  criterion = []
  min_samples_leaf = []
  min_samples_split = []
  n_estimators = []

  for index, trial in enumerate(valid_trial_list):
    vals_trial = space_eval(space_rf, unpack_values(trial))
    
    indexes.append(index)
    criterion.append(vals_trial['criterion'])
    min_samples_leaf.append(vals_trial['min_samples_leaf'])
    min_samples_split.append(vals_trial['min_samples_split'])
    n_estimators.append(vals_trial['n_estimators'])
  
  data_df = {'index': indexes,
  'criterion': criterion, 
  'min_samples_leaf': min_samples_leaf,
  'min_samples_split': min_samples_split,
  'n_estimators': n_estimators,
  'loss': losses}

  df = pd.DataFrame(data=data_df)
  df.to_csv("hpo_rf.csv", index=False)

export_data(trials)
