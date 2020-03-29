import csv 
import sys
import matplotlib.pyplot as plt
import numpy as np

PATH = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/eval/'

colour_dict = {
    'sphero_blue' : 0,
    'sphero_dark_blue' : 0,
    'sphero_dark_green' : 0,
    'sphero_green' : 0,
    'sphero_light_blue' : 0,
    'sphero_lime_green' : 0,
    'sphero_magenta' : 0,
    'sphero_purple' : 0,
    'sphero_red' : 0,
    'sphero_yellow' : 0 
}  

detect_dict = {}

number = 0
line_counter = 0

with open('/home/josi/OvGU/Rolling Swarm/rs_nn_training/eval/mit_nachhelfen/evaldetectionfile_light_blue_4x_step19.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    
    for csv_row in reader:
        if line_counter != 0 and csv_row[2] != "no detection": 
            number += 1
            colour_dict[csv_row[3]] += 1
        line_counter += 1

print("results" +  str(dict(colour_dict))) 

with open(PATH + sys.argv[1] + '.csv', 'w') as f:
    w = csv.DictWriter(f, colour_dict.keys())
    w.writeheader()
    w.writerow(colour_dict)

if (len(sys.argv) > 1):
    expected_colour = str(sys.argv[1])   
    detections = colour_dict[expected_colour]

    detection_ratio = detections / number

    print(detection_ratio)

colors_all = ['blue', 'darkblue', 'darkgreen', 'green', 'lightblue', 'limegreen', 'magenta', 'purple', 'red', 'yellow']

i = 0
for k,v in dict(colour_dict).items():
    if int(v) > 3:
        detect_dict[colors_all[i]] = v / number
    i += 1

colors = []
for k in dict(detect_dict).keys():
    colors.append(k)

plt.bar(range(len(detect_dict)), list(detect_dict.values()), align='center', width=0.2, color=colors, alpha=0.8)

plt.xticks(range(len(detect_dict)), list(detect_dict.keys()))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.gca().yaxis.grid(True)
plt.legend(frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

#plt.title('Farbverteilung')
plt.xlabel('erkannte Farben')
plt.ylabel('Anteil')

plt.show()


            




            
        
