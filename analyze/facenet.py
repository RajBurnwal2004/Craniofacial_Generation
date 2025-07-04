from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os
import torch
import numpy as np
from PIL import Image

# Initialize MTCNN for face detection and alignment
mtcnn = MTCNN(image_size=160, margin=14)

# Load pre-trained FaceNet (InceptionResnetV1)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Folder of generated images
image_folder = 'eval/fake_B'
embeddings = []
image_names = []

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, filename)
    img = Image.open(img_path).convert('RGB')

    # Detect and align face
    face = mtcnn(img)
    if face is None:
        print(f"[WARN] No face detected in {filename}")
        continue

    # Generate embedding
    face_embedding = resnet(face.unsqueeze(0))  # Shape: (1, 512)
    embeddings.append(face_embedding.detach().numpy()[0])
    image_names.append(filename)
    print(f"[OK] {filename}: Face embedding extracted.")

# Convert embeddings to numpy
embeddings = np.array(embeddings)
# After computing embeddings and image_names...

# Save to disk
np.save('generated_embeddings.npy', embeddings)
with open('image_names.txt', 'w') as f:
    for name in image_names:
        f.write(name + '\n')

print("✅ Embeddings and image names saved.")
