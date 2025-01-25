# T-SNE Visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def visualization_with_tsne(feature, labels, model_name):
    tsne = TSNE(n_components=2, random_state=42, n_jobs=16)
    reduce_features = tsne.fit_transform(feature)

    plt.figure(figsize=(10, 10))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduce_features[idx, 0], reduce_features[idx, 1], label=f"Class {label}", alpha=0.6)
    plt.title(f't-SNE Visualization of {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{model_name}_tsne.png")
    plt.show()
