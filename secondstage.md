# Second Stage

neue Farben hinzufügen:
ros_ws/src/Rolling_Swarm/sphero_node/nodes/sphero.py

nn_tracking_node.h $\rightarrow$ farben

nn_tracking_node.cpp -> Anzahl erhöhen



## to do 
- RecordInspector -> Anzahl Daten pro Grad 
- unterschiede experimenttypen
- bin vs reg
- copter und youbot entfernen
- was macht SecondStage.py mit testdaten
- quality
- undefined
- custom objects - categoricalcrossentropy & accuracy
- cropping?
- PredictionInspection deprecated


## Vorbereitung

Packages:
- tqdm
- keras
- lxml
- sklearn (pip3 install scipy, pip3 install sklearn)
- pandas


Directory:  
dir  
|-- object_detection (tensorflow)  
|-- rs_nn_training  
|-- |-- Second Stage  
|-- data  
|-- |-- train (Trainingsdatensatz)  
|-- |-- test (Testdatensatz)  
|-- output  
|-- |-- inference  
|-- |-- second_stage (h5 Modelle)


[Manual protobuf-compiler installation and usage] (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage)
sonst fehlt string_int_label_map_pb2 in object_detection


| file | description |
| ---- | -------------- |
| convertPNGsToTFRecords | images und label von compositor in tfrecords konvertieren |
| exp_def | Konfiguration Netz |
| label_map | Definitonen der Label IDs |
| PredictionInspection | Evaluierung zeigt welche images falsch klassifiziert wurden  |
| SecondStage | Training Netz |
| second_stage_utils | keras custom objects, load images |
| TFRecordInspectorSStage | Anzahl Daten pro Klasse |


## Training

SecondStage.py -e "exp_name"+"type"  
	"exp_name" = Name des Experiments aus exp_def  
	"type" = reg | cat | bin  
	reg = orientation  
	cat = Klassifikation Farben  
	bin = ?  

Bsp.: -e sstage_default_sphero_cat


## Testen

PredicitonInspection.py

TRAIN_OR_VAL: "train" | "val"
MODE: "regression" | "classification"







