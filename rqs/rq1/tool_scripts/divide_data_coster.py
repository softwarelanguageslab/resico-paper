import pandas as pd
from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
	FILE = "contextsCOSTER_strat.csv"
	FOLDER_STRAT = "strat_coster"

	print("Reading data file ...")
	data = pd.read_csv(f"{FILE}")
	print("Done!")

	print("Defining the X and y from the data ...")
	X = data[data.columns[:-1]]
	y = data[["FQN"]]
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
		train_df.to_csv(f"{FOLDER_STRAT}/train_{i}.csv", index=False)
		test_df.to_csv(f"{FOLDER_STRAT}/test_{i}.csv", index=False)
	print("Done!")
