import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# Folder containing the generated faces
image_folder = './eval/fake_B'

# Number of images to show
K = 5

# Sort indices
sorted_indices = np.argsort(average_sim_per_fake)
top_k = sorted_indices[-K:][::-1]   # Highest similarity
bottom_k = sorted_indices[:K]       # Lowest similarity

def show_image_grid(indices, title, sim_scores):
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        img_path = os.path.join(image_folder, fake_names[idx])
        img = Image.open(img_path).convert('RGB')
        plt.subplot(1, K, i + 1)
        plt.imshow(img)
        plt.title(f"{fake_names[idx]}\n{sim_scores[idx]:.2f}")
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Show grids
show_image_grid(top_k, "🎯 Most Realistic (Top 5)", average_sim_per_fake)
show_image_grid(bottom_k, "🚫 Least Realistic (Bottom 5)", average_sim_per_fake)
