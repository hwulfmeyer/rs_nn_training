import matplotlib.pyplot as plt
import numpy as np

colour_dict = {
    'blue' : 0.903119376124775,
    'dark_blue' : 0.9031948881789137,
    'dark_green' : 0.9989550679205852,
    'green' : 0.760806916426513,
    'light_blue' : 0.7081081081081081,
    'lime_green' : 0.6460807600950119,
    'magenta' : 0.9538142189932538,
    'purple' : 0.7074468085106383,
    'red' : 0.8997429305912596,
    'yellow' : 0.787359716479622,
} 

colors = ['blue', 'darkblue', 'darkgreen', 'green', 'lightblue', 'limegreen', 'magenta', 'purple', 'red', 'yellow']
#colors = ['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'lightcoral', 'lightcoral', 'lightcoral', 'cornflowerblue', 'lightcoral', 'cornflowerblue', 'lightcoral']
plt.bar(range(len(colour_dict)), list(colour_dict.values()), align='center', width=0.3, color=colors, alpha=0.8)
plt.xticks(range(len(colour_dict)), list(colour_dict.keys()))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.gca().yaxis.grid(True)
plt.legend(frameon=False)

plt.title('Recall')
plt.xlabel('colour sphero')
plt.ylabel('Recall')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

#ax.spines['top'].set_visible(False)

plt.show()
