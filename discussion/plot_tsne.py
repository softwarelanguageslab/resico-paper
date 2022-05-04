import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = pd.read_csv("selected_data.csv", header=None)

X = data.to_numpy()[:, :100]
y = data[data.columns[-1]].to_numpy()

print("Shapes of the selected data:")
print(X.shape, y.shape)

feat_cols = ['feature_' + str(i) for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y

print('Size of the dataframe: {}'.format(df.shape))

tsne = TSNE(verbose=1)
tsne_results = tsne.fit_transform(X)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(13,7))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 14),
    data=df,
    legend=False,
    size=30
)

x_coord = df['tsne-2d-one'].to_numpy()
y_coord = df['tsne-2d-two'].to_numpy()

for (i, (x_point, y_point)) in enumerate(zip(x_coord, y_coord)):
    plt.text(x = x_point + 0.4, y = y_point + 0.4, s = y[i],
        fontdict=dict(size=11)
    )

plt.box(False)
plt.axis('off')
plt.show()
