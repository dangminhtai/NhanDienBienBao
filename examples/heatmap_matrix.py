import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Tạo ma trận ví dụ
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Vẽ heatmap
sns.heatmap(matrix, annot=True, cmap="YlGnBu")  # annot=True để hiển thị giá trị
plt.show()