import matplotlib.pyplot as plt

plt.hist(average_sim_per_fake, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Fake-to-Real Face Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Fake Images")
plt.grid(True)
plt.show()
