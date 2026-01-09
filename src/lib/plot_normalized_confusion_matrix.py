import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_normalized_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    
    # Normalize the confusion matrix row-wise
    cm_percentage = cm / np.sum(cm, axis=1, keepdims=True) * 100
    
    # Create the annotation array
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]} \n({cm_percentage[i, j]:.2f}%)'

    sns.heatmap(cm_percentage, annot=annot, fmt='', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()