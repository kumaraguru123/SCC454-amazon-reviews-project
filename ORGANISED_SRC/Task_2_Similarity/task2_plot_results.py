import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("task2_benchmark_results.csv")

print(df)

methods = df["method"].unique()

plt.figure(figsize=(10,6))

for m in methods:
    subset = df[df["method"] == m]

    if subset["dataset_size"].dtype != object:
        plt.plot(
            subset["dataset_size"],
            subset["time_sec"],
            marker="o",
            label=m
        )

plt.xlabel("Dataset Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Similarity Algorithm Performance Comparison")

plt.legend()
plt.grid(True)

plt.savefig("task2_similarity_performance.png")

plt.show()