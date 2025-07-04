import matplotlib.pyplot as plt

fid_log_path = "fid_scores.txt"

epochs = []
fids = []

with open(fid_log_path, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("Epoch") and "FID" in line:
        try:
            epoch = int(line.split()[1])
            # FID value is two lines after "Epoch X - FID: ..."
            fid_line = lines[i + 3].strip()
            fid = float(fid_line)
            epochs.append(epoch)
            fids.append(fid)
            i += 4  # skip ahead
        except:
            i += 1
    else:
        i += 1

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, fids, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6)
plt.title("FID vs Epoch (CycleGAN on Skull2Face)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.grid(True)
plt.xticks(epochs, rotation=45)
plt.tight_layout()
plt.savefig("fid_vs_epoch_clean.png")
plt.show()
