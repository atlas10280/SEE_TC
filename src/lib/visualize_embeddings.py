# define umap/tsne handling of embedded features
import umap
import matplotlib.pyplot as plt
import numpy as np


def visualize_embeddings(embeddings, labels, method='umap', manual_colors = ['#FF0000', '#000000', '#990066', '#999999', '#006600', '#6600CC']):
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    # Check if there are enough colors
    if len(unique_labels) > len(manual_colors):
        raise ValueError(f"Not enough manual colors defined for {len(unique_labels)} labels.")

    # Create mapping using only as many colors as needed
    label_to_color = {label: manual_colors[i] for i, label in enumerate(unique_labels)}

    # Dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=710)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    reduced = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(7, 7))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], color=label_to_color[label],
                    label=f'Class {label}', s=10, alpha=0.8)

    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=2, loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()
    
    return(reduced) # return the coordinates so we can inspect cells
