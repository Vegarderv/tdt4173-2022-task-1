import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import k_means as km # <-- Your implementation

sns.set_style('darkgrid')

data_1 = pd.read_csv('k_means\data_1.csv')

plt.figure(figsize=(5, 5))
sns.scatterplot(x='x0', y='x1', data=data_1)
plt.xlim(0, 1); plt.ylim(0, 1)

# Fit Model 
X = data_1[['x0', 'x1']]
model_1 = km.KMeans(accuracy = 0.001) # <-- Should work with default constructor  
model_1.fit(X)
# Compute Silhouette Score 
z = model_1.predict(X)
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')
print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')

# Plot cluster assignments
C = model_1.get_centroids()
K = len(C)
_, ax = plt.subplots(figsize=(5, 5), dpi=100)
sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax)
sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
ax.legend().remove()