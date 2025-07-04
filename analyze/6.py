import matplotlib.pyplot as plt

# Second set of FID values (from your image)
epoch_fid_dict = {
    100: 72.07,
    300: 41.61,
    500: 32.42,
    700: 29.60,
    900: 32.03,
    1100: 18.94,
    1300: 19.64,
    1500: 14.88,
    1700: 15.47,
    1900: 12.94,
    2100: 14.33,
    2300: 12.02,
    2500: 12.00,
    2700: 12.10,
    2900: 11.87,
    3100: 13.15,
    3300: 12.07,
    3500: 11.85,
    3700: 12.27,
    3900: 18.04,
    4100: 11.70,
    4300: 14.44,
    4500: 11.44
}

# Sort the dictionary by epoch
sorted_epochs = sorted(epoch_fid_dict.keys())
sorted_fids = [epoch_fid_dict[ep] for ep in sorted_epochs]

# Plotting without coordinate annotations
plt.figure(figsize=(12, 6))
plt.plot(sorted_epochs, sorted_fids, marker='o', linestyle='-', color='green')
plt.title('FID Score vs Epoch (SRC Loss)')
plt.xlabel('Epoch')
plt.ylabel('FID Score')
plt.grid(True)

plt.tight_layout()
plt.show()

