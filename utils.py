import numpy as np
import tensorflow as tf

def splitter(x):
    row = np.array(x.split(',')).astype(np.float32)
    label = row[0]
    pixel = row[1:]
    return label, pixel


def normalize(label, pixel):
    """Returns a normalized feature array between 0 and 1"""
    pixel = pixel.astype(np.float32)
    #return label, pixel/np.max(pixel)
    return label, (pixel - np.min(pixel))/(np.max(pixel) - np.min(pixel))

def decode(line, default_values):
	item = tf.decode_csv(line, default_values)
	return item[0], item[1:]