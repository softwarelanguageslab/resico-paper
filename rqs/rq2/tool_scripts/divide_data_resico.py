from email import header
import pandas as pd
from sklearn.model_selection import StratifiedKFold

n_row = 50
path_data = "sample_{0}".format(n_row)

FILE = "{}/data.csv".format(path_data)
FOLDER_STRAT = "{}/strat".format(path_data)

print("Reading data file ...")
data = pd.read_csv(f"{FILE}", header=None)
print("Done!")

print("Defining the X and y from the data ...")
X = data[data.columns[:-1]].to_numpy()
y = data[data.columns[-1]].to_numpy()
print("Done!")

train_indices = list()
test_indices = list()

print("Dividing data ...")
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X, y)
for train, test in folds:
	train_indices.append(train)
	test_indices.append(test)

train_dfs = list()
test_dfs = list()

for i in range(10):
	train_dfs.append(data.iloc[list(train_indices[i]), :])
	test_dfs.append(data.iloc[list(test_indices[i]), :])
print("Done!")

print("Saving divided data ...")
for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
	print(f"Fold {i + 1} / 10 ...")
	train_df.to_csv(f"{FOLDER_STRAT}/train_{i}.csv", index=False, header=False)
	test_df.to_csv(f"{FOLDER_STRAT}/test_{i}.csv", index=False, header=False)
print("Done!")
