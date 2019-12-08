import tensorflow as tf
from itertools import zip_longest
from collections import OrderedDict

def readTfEvents(file):
    tensorboard_dict = {}
    tensorboard_dict['Epoch'] = []
    # See https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/util/event.proto
    # and https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/framework/summary.proto
    for e in tf.train.summary_iterator(file):
        if len(tensorboard_dict['Epoch']) == 0:
            tensorboard_dict['Epoch'].append(e.step)
        if tensorboard_dict['Epoch'][-1] != e.step:
            tensorboard_dict['Epoch'].append(e.step)
        for v in e.summary.value:
            if v.tag not in tensorboard_dict:
                tensorboard_dict[v.tag] = []
            tensorboard_dict[v.tag].append(v.simple_value)
    tensorboard_dict = OrderedDict(sorted(tensorboard_dict.items()))
    return tensorboard_dict

def filterRange(tensorboard_dict, epoch_range = range(20000, 25000)):
    filtered = {}
    for epoch,*values in zip(tensorboard_dict['Epoch'],*tensorboard_dict.values()):
        if epoch in epoch_range:
            for i,k in enumerate(tensorboard_dict.keys()):
                if k not in filtered:
                    filtered[k] = []
                filtered[k].append(values[i])
    return filtered
