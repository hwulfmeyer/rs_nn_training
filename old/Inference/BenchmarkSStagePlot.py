import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

from CNNRobotLocalisation.Utils.file_utils import *

GPU_FILE = '/home/lhoyer/cnn_robot_localization/benchmark/benchmark-result-lukas-pc-gpu-1080.csv'
CPU_FILE = '/home/lhoyer/cnn_robot_localization/benchmark/benchmark-result-lukas-pc-cpu.csv'
OUT = '/home/lukas/Nextcloud/Studium/Bachelorarbeit/Overleaf/Plots/benchmark_sstage_2.pdf'

# See: https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
# print(plt.style.available)
mpl.style.use('classic')
mpl.rcParams['lines.linewidth'] = 2.0
# mpl.rcParams.update({'font.size': 14})
# mpl.rcParams["font.family"] = "Times New Roman"
black_cycle = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':','-.']))
red_cycle = (cycler('color', ['r']) * cycler('linestyle', ['-', '--', ':','-.']))
# plt.rcParams['image.cmap'] = 'gray'
plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))


# fig, ax = plt.subplots(1,1)
# rows = readCSV(CPU_FILE)
# cols = zip(*rows)
# ax.set_prop_cycle(black_cycle)
# for col in cols:
#     ax.plot(range(1,len(col)),[float(i) for i in col[1:]], label='CPU ' + str(col[0]).replace('Alpha ',r'$\alpha$='))
# rows = readCSV(GPU_FILE)
# cols = zip(*rows)
# ax.set_prop_cycle(red_cycle)
# for col in cols:
#     ax.plot(range(1,len(col)),[float(i) for i in col[1:]], label='GPU ' + str(col[0]).replace('Alpha ',r'$\alpha$='))
# plt.ylabel('Execution time in ms')
# plt.xlabel('Number of input crops')
# #plt.gca().set_xlim([1,20])
# #plt.gca().set_ylim([0,35])
# plt.legend(loc='upper left',ncol=2)
# # plt.title('Inference times of second stage mobilenet')
# plt.grid()
# ax = plt.gca()
# ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
# plt.gca().set_aspect(0.5*ratio_default)
# plt.savefig(OUT, bbox_inches='tight')
# #plt.show()

aspect_ratio = 1.08
fig, (ax, ax2) = plt.subplots(1,2, sharex=True)
rows = readCSV(CPU_FILE)
cols = zip(*rows)
ax.set_prop_cycle(black_cycle)
for col in cols:
    ax.plot(range(1,len(col)),[float(i) for i in col[1:]], label=str(col[0]).replace('Alpha ',r'$\alpha$='))
#ax.set_xlabel('Number of input crops', ha='center')
fig.text(0.5, 0.171, 'Number of input crops', ha='center')
ax.set_ylabel('Execution time in ms')
ax.legend(loc='upper left')
ax.grid()
ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
ax.set_aspect(aspect_ratio*ratio_default)
ax.set_title("Intel Xeon E3-1230 v3")


rows = readCSV(GPU_FILE)
cols = zip(*rows)
ax2.set_prop_cycle(black_cycle)
for col in cols:
    ax2.plot(range(1,len(col)),[float(i) for i in col[1:]], label=str(col[0]).replace('Alpha ',r'$\alpha$='))
#ax2.set_xlabel('Number of input crops')
#ax2.set_ylabel('Execution time in ms')
#plt.legend(loc='upper left')
ax2.grid()
ratio_default=(ax2.get_xlim()[1]-ax2.get_xlim()[0])/(ax2.get_ylim()[1]-ax2.get_ylim()[0])
ax2.set_aspect(aspect_ratio*ratio_default)
ax2.set_title("Nvidia GeForce GTX 1080")


plt.savefig(OUT, bbox_inches='tight')
#plt.show()
