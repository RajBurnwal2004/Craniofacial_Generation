import os
import subprocess
import matplotlib.pyplot as plt
import glob
import shutil
from cleanfid import fid

# --------- CONFIGURABLE PARAMETERS ---------
start_epoch = 100
end_epoch = 4600
step = 200

dataroot = "./datasets/skull2face"
model_name = "skull2face_final"
direction = "BtoA"  # "BtoA" for other direction

eval_real_path = "./eval/real_A"
eval_fake_path = "./eval/fake_A"

fid_save_file = f"fid_scores_{direction}.txt"
plot_save_file = f"fid_plot_{direction}.png"

# --------- UTILITY FUNCTIONS ---------
def clear_folder(folder_path):
    files = glob.glob(f"{folder_path}/*")
    for f in files:
        os.remove(f)

def copy_images(src_folder, dst_folder, suffix):
    os.makedirs(dst_folder, exist_ok=True)
    for file_name in os.listdir(src_folder):
        if file_name.endswith(suffix):
            shutil.copy(os.path.join(src_folder, file_name), os.path.join(dst_folder, file_name))

# --------- CLEAR OLD FID FILE IF EXISTS ---------
open(fid_save_file, "w").close()

# --------- LOOP THROUGH EPOCHS ---------
fid_scores = []

for epoch in range(start_epoch, end_epoch + 1, step):
    print(f"\n===== Running test for epoch {epoch} =====\n")

    # 1. Run test
    subprocess.run([
        "python", "test.py",
        "--dataroot", dataroot,
        "--name", model_name,
        "--model", "cycle_gan",
        "--epoch", str(epoch),
        "--gpu_ids", "0",
        "--num_test", "14",  # Adjust if needed
        "--direction", direction,
    ])

    # 2. Copy fake_B images for FID
    result_path = f"./results/{model_name}/test_{epoch}/images"
    clear_folder(eval_fake_path)
    copy_images(result_path, eval_fake_path, "_fake_A.png")

    # 3. Compute FID
    fid_score = fid.compute_fid(eval_real_path, eval_fake_path, device="cuda")
    fid_scores.append((epoch, fid_score))

    # 4. Save FID score
    with open(fid_save_file, "a") as f:
        f.write(f"Epoch {epoch} - FID: {fid_score:.2f}\n")

# --------- PLOT FID vs. EPOCHS ---------
epochs, scores = zip(*fid_scores)

plt.figure(figsize=(10, 5))
plt.plot(epochs, scores, marker='o', linestyle='-', color='navy')
plt.title(f"FID vs. Epochs ({direction})", fontname="Times New Roman")
plt.xlabel("Epoch", fontname="Times New Roman")
plt.ylabel("FID Score", fontname="Times New Roman")
plt.grid(True)
plt.xticks(epochs)
plt.savefig(plot_save_file)
plt.show()

print(f"\n✅ Done! FID scores saved to {fid_save_file}")
print(f"📊 Plot saved as {plot_save_file}")

