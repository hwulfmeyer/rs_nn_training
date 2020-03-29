import csv 

max_diff = 0
min_diff = 1
avg_diff = 0

num_logs = 0

with open('/home/josi/OvGU/Rolling Swarm/rs_nn_training/eval/evaldetectionfile_zeit_neu.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
        
    for csv_row in reader:
        diff = float(csv_row[1]) - float(csv_row[0])
        if diff > max_diff:
            max_diff = diff
        if num_logs > 0 and diff < min_diff:
            min_diff = diff 
        avg_diff += diff
        num_logs += 1

avg_diff /= num_logs

print("max: " + str(max_diff))
print("min: " + str(min_diff))
print("avg: " + str(avg_diff))
