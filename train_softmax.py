#!/usr/bin env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import argparse
from utils import splitter, normalize
from model import graph


def main(args):
    bs = args.batch_size
    trainpath = args.train_csv
    validationpath = args.validation_csv
    learning_rate = args.lr
    default_values = [[0.0] for _ in range(785)]
    
    def decode(line):
        item = tf.decode_csv(line, default_values)
        return item[0], item[1:]
    
    
    # Skip the header and filter any comments.
    training_dataset = tf.data.TextLineDataset(trainpath).skip(1).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))
    validation_dataset = tf.data.TextLineDataset(validationpath).skip(1).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))
    
    # The dataset api reads the csv as text.
    # Using the below function we can split the text into labels and pixels.
    
    training_dataset = (training_dataset.cache().map(decode))
    #training_dataset = training_dataset.map(
    #    lambda x: tf.py_func(splitter, [x], [tf.float32, tf.float32]))
    validation_dataset = validation_dataset.cache().map(decode)
    
    # Normalize the dataset to 0-1 range
    training_dataset = training_dataset.map(
        lambda label, pixel: tf.py_func(normalize, [label, pixel], [tf.float32, tf.float32]))
    validation_dataset = validation_dataset.map(
        lambda label, pixel: tf.py_func(normalize, [label, pixel], [tf.float32, tf.float32]))
    
    # Randomly shuffles the dataset
    training_dataset = training_dataset.shuffle(buffer_size=args.buffer_size)
    
    # Creating batchs here for training
    training_dataset = training_dataset.batch(bs)
    validation_dataset = validation_dataset.batch(bs)
    
    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    
    next_element = iterator.get_next()
    
    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_initializable_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()
    
    # Define training op here
    #train_graph = tf.Graph()
    #tf.reset_default_graph()
    
    x = tf.placeholder('float32',shape=[bs,None])
    y = tf.placeholder('int32',shape=[bs])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    logits = graph(x) # training mode
    _y = tf.cast(tf.one_hot(y, depth=10), tf.int32)
    loss = create_loss(logits, _y)
    train_op, global_step = create_optimizer(logits, learning_rate=learning_rate)
    training_loss = []
    epoch_loss = []
    
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), 
                          tf.local_variables_initializer()))
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        
        for i in range(args.nrof_epochs):
            sess.run(training_iterator.initializer)
            while True:
                try:
                    label_batch, image_batch = sess.run(next_element, feed_dict={handle: training_handle})
                    _, _loss, g = sess.run([train_op, loss, global_step], feed_dict = {x:image_batch, y:label_batch})
                    training_loss.append(_loss)
                    epoch_loss.append(_loss)
                    if tf.train.global_step(sess, global_step)%10==0:
                        print_results(g, training_loss)
                except tf.errors.OutOfRangeError:
                    print('Out of Data, Training Finished!')
                    break
                #finally:
                #    sess.run(validation_iterator.initializer)
                #    sess.run(next_element, feed_dict={handle: validation_handle})
            print('Epoch: ',i, end=' ')
            print_results(i, epoch_loss)
            epoch_loss = []
                #sess.run(validation_iterator.initializer)
                #sess.run(next_element, feed_dict={handle: validation_handle})
    return True
    

def print_results(iteration, losses):
    print("iteration: {0:5d} loss: {1:0.3f}"
          .format(iteration, np.mean(losses)))


def create_accuracy(logits, outputs):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(outputs, 1))
    print(correct_prediction)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return accuracy
    

def create_loss(logits, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    return loss


def create_optimizer(loss, learning_rate):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    return train_op, global_step

    
def parser(argv):
    parser = argparse.ArgumentParser(description='Trains a Deep Neural Network on Fashion MNIST Data')
    parser.add_argument('--train_csv', default='training.csv', type=str, required=True, help='Path to the training csv.')
    parser.add_argument('--validation_csv', default='validation.csv', type=str, help='Path to the validation csv.')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch Size of one iteration.')
    parser.add_argument('--buffer_size', default=10000, type=int, help='Buffer Size for random selection of images.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate.')
    parser.add_argument('--nrof_epochs', default=20, type=int, help='Number of Epochs for training.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parser(sys.argv[1:]))