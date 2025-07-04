from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
with open('fake_names.txt', 'r') as f:
    fake_names = [line.strip() for line in f]

with open('real_names.txt', 'r') as f:
    real_names = [line.strip() for line in f]

real_embeds = np.load('real_embeddings.npy')
fake_embeds = np.load('fake_embeddings.npy')

# Compute cosine similarity of each fake to all real
similarity_scores = cosine_similarity(fake_embeds, real_embeds)

# Average similarity per fake image
average_sim_per_fake = similarity_scores.mean(axis=1)

# Overall realism score
mean_similarity = np.mean(average_sim_per_fake)

# Show top and bottom examples
top_k = np.argsort(average_sim_per_fake)[-5:]
bottom_k = np.argsort(average_sim_per_fake)[:5]

print(f"\n✅ Mean Similarity (Realism Score): {mean_similarity:.4f}")
print("\n🎯 Most Realistic (Top 5):")
for idx in reversed(top_k):
    print(f"{fake_names[idx]} - similarity: {average_sim_per_fake[idx]:.4f}")

print("\n🚫 Least Realistic (Bottom 5):")
for idx in bottom_k:
    print(f"{fake_names[idx]} - similarity: {average_sim_per_fake[idx]:.4f}")

import matplotlib.pyplot as plt

plt.hist(average_sim_per_fake, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Fake-to-Real Face Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Fake Images")
plt.grid(True)
plt.show()

import csv
import numpy as np

# Assuming `average_sim_per_fake` and `fake_names` are already defined

with open('face_similarity_scores.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'SimilarityScore'])

    for name, score in zip(fake_names, average_sim_per_fake):
        writer.writerow([name, score])

print("✅ CSV saved as 'face_similarity_scores.csv'")

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
