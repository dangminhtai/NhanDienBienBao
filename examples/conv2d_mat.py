import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ma trận input 4x4
input_matrix = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
])

# Kernel 2x2
kernel = np.array([
    [1, 0],
    [0, 1]
])

# Kích thước output
H_out = input_matrix.shape[0] - kernel.shape[0] + 1
W_out = input_matrix.shape[1] - kernel.shape[1] + 1
output = np.zeros((H_out, W_out))

# Tích chập (Conv2d)
for i in range(H_out):
    for j in range(W_out):
        region = input_matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]]
        output[i, j] = np.sum(region * kernel)

# Vẽ heatmap
sns.heatmap(output, annot=True, cmap="YlOrRd")
plt.title("Feature Map after Conv2d")
plt.show()