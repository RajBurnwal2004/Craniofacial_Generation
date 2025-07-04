from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import os

mtcnn = MTCNN(image_size=160, margin=14)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embeddings_from_folder(folder_path):
    embeddings = []
    valid_images = []

    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(path).convert('RGB')
            face = mtcnn(img)
            if face is None:
                print(f"[WARN] No face in {img_name}")
                continue
            emb = resnet(face.unsqueeze(0))
            embeddings.append(emb.detach().numpy()[0])
            valid_images.append(img_name)
        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")

    return np.array(embeddings), valid_images

# Replace these with your actual folders
real_embeds, real_names = get_embeddings_from_folder('./eval/real_B')
fake_embeds, fake_names = get_embeddings_from_folder('./eval/fake_B')
with open('fake_names.txt', 'w') as f:
    for name in fake_names:
        f.write(name + '\n')

with open('real_names.txt', 'w') as f:
    for name in real_names:
        f.write(name + '\n')

np.save('real_embeddings.npy', real_embeds)
np.save('fake_embeddings.npy', fake_embeds)
