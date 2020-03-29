import csv 

colour_dict = {
    'blue' : 0,
    'dark_blue' : 0,
    'dark_green' : 0,
    'green' : 0,
    'light_blue' : 0,
    'lime_green' : 0,
    'magenta' : 0,
    'purple' : 0,
    'red' : 0,
    'yellow' : 0,
} 

pred = []

with open('/home/josi/OvGU/Rolling Swarm/rs_nn_training/eval/csv/all_detections.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')

    n = 0
    for k in dict(colour_dict).keys():
        tp = 0
        p = 0
        row = 0
        for csv_row in reader:
            if row == n:
                tp = int(csv_row[n])
            p += int(csv_row[n])
            print("fp: " + str(p))
            row += 1
        print(tp)
        print(p)
        if p > 0 and tp > 0:
            precision = tp / p
        print(precision)
        colour_dict[k] = precision
        n += 1
        print(n)

print("results" +  str(dict(colour_dict))) 

########################################################################################

plt.bar(range(len(detect_dict)), list(detect_dict.values()), align='center', width=0.2, color=colors, alpha=0.8)

plt.xticks(range(len(detect_dict)), list(detect_dict.keys()))
plt.yticks(np.arange(0, 1, step=0.1))
plt.gca().yaxis.grid(True)
plt.legend(frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.xlabel('detected colours')
plt.ylabel('percentage')

plt.show()
        


