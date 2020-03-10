import matplotlib.pyplot as plt
import numpy as np

colour_dict = {
    'blue' : 0.6921839080,
    'dark_blue' : 0.9121471614,
    'dark_green' : 0.5861434703,
    'green' : 0.973371569,
    'light_blue' : 0.79393939393,
    'lime_green' : 0.90066225165,
    'magenta' : 0.9014222658,
    'purple' : 0.9795417349,
    'red' : 0.985915492957,
    'yellow' : 0.86166774402,
} 

colors = ['blue', 'darkblue', 'darkgreen', 'green', 'lightblue', 'limegreen', 'magenta', 'purple', 'red', 'yellow']
#colors = ['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'lightcoral', 'lightcoral', 'lightcoral', 'cornflowerblue', 'lightcoral', 'cornflowerblue', 'lightcoral']
plt.bar(range(len(colour_dict)), list(colour_dict.values()), align='center', width=0.3, color=colors, alpha=0.8)
plt.xticks(range(len(colour_dict)), list(colour_dict.keys()))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.gca().yaxis.grid(True)
plt.legend(frameon=False)

plt.title('Precision')
plt.xlabel('colour sphero')
plt.ylabel('Precision')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

#ax.spines['top'].set_visible(False)

plt.show()


