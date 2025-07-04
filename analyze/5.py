import matplotlib.pyplot as plt

# First set of FID values (from first image)
epoch_fid_dict = {
    100: 307.50,
    300: 222.54,
    500: 227.97,
    700: 199.26,
    900: 172.99,
    1100: 159.88,
    1300: 173.55,
    1500: 163.46,
    1700: 152.04,
    1900: 146.52,
    2100: 148.88,
    2300: 143.63,
    2500: 150.90,
    2700: 146.36,
    2900: 143.88,
    3100: 139.52,
    3300: 144.09,
    3500: 146.52,
    3700: 148.44,
    3900: 147.85,
    4100: 151.54,
    4300: 153.65,
    4500: 155.08
}

# Sort the dictionary by epoch
sorted_epochs = sorted(epoch_fid_dict.keys())
sorted_fids = [epoch_fid_dict[ep] for ep in sorted_epochs]

# Plotting without coordinate annotations
plt.figure(figsize=(12, 6))
plt.plot(sorted_epochs, sorted_fids, marker='o', linestyle='-', color='blue')
plt.title('FID Score vs Epoch (Vanilla CycleGAN)')
plt.xlabel('Epoch')
plt.ylabel('FID Score')
plt.grid(True)

plt.tight_layout()
plt.show()

