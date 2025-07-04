import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved data
embeddings = np.load('generated_embeddings.npy')
with open('image_names.txt', 'r') as f:
    image_names = [line.strip() for line in f]

# Compute cosine similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, xticklabels=image_names, yticklabels=image_names, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Cosine Similarity between Generated Face Embeddings")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
