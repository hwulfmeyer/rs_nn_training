import matplotlib.pyplot as plt
import numpy as np

heatmap = np.load("heatmap")
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.show()
