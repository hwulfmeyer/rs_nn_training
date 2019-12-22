import os, csv, json, pickle
from lxml import etree
from object_detection.utils import dataset_util

def get_recursive_file_list(path, file_matchers = None,
                            file_excludes = None,
                            file_extensions = None,
                            sort = True):
    print("#############################") 
    file_list = []
    for root, dirs, files in os.walk(path, ):
        print('------------------')
        print(root)
        print(dirs)
        print('------------------')
        for f in files:
            file = root+'/'+f
            if (file_matchers == None or any(m in file for m in file_matchers)) and \
               (file_excludes == None or not any(m in file for m in file_excludes)) and \
               (file_extensions == None or file.endswith(tuple(file_extensions))):
                file_list.append(file)
    print("#############################")  
    exit()  
    print(file_list)    
    if sort:
        file_list.sort()
    return file_list

def writeCSV(file, rows):
    with open(file, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            spamwriter.writerow(row)

def readCSV(file):
    rows = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    return rows

def save_pkl(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def parseXML(file):
    #with tf.gfile.GFile(file, 'r') as fid:
    #    xml_str = fid.read()
    with open(file, 'r') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    return dataset_util.recursive_parse_xml_to_dict(xml)
