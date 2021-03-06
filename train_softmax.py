#!/usr/bin env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import argparse
from utils import splitter, normalize
from model import graph

# TO DO
# Add saver
# Add summary
# Add validation after every epoch
# Add Validation accuracy


def main(args):
    bs = args.batch_size
    trainpath = args.train_csv
    validationpath = args.validation_csv
    learning_rate = args.lr
    logdir = args.log_dir

    if not os.path.exists(logdir):
        os.makedirs(logdir)

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
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    logits = graph(x) # training mode

    _y = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)
    #loss = create_loss(logits, _y)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits)
    loss = tf.reduce_mean(loss,axis=0)
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    #train_op, global_step = create_optimizer(logits, learning_rate=learning_rate)
    training_loss = []
    epoch_loss = []
    
    with tf.Session() as sess:
        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logdir, sess.graph)

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
                    #np.save('x.npy',image_batch)
                    #print(label_batch)
                    # xx = sess.run([_y], feed_dict = {y:label_batch})
                    summary, _loss,_, g = sess.run([merged, loss,train_op,global_step], feed_dict = {x:image_batch, y:label_batch})

                    training_loss.append(_loss)
                    epoch_loss.append(_loss)
                    if tf.train.global_step(sess, global_step)%10==0:
                        train_writer.add_summary(summary)
                        print_results(g, training_loss)
                        training_loss = []
                except tf.errors.OutOfRangeError:
                    print('Out of Data, Training Finished!')
                    break
                #finally:
                #    sess.run(validation_iterator.initializer)
                #    sess.run(next_element, feed_dict={handle: validation_handle})
            print_results_epoch(i, epoch_loss)
            epoch_loss = []
                #sess.run(validation_iterator.initializer)
                #sess.run(next_element, feed_dict={handle: validation_handle})
        train_writer.close()
    return True
    

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def print_results(iteration, losses):
    print("Batch: {0:5d} loss: {1:0.3f}"
          .format(iteration, np.mean(losses)))


def print_results_epoch(iteration, losses):
    print("Epoch: {0:5d} loss: {1:0.3f}"
          .format(iteration+1, np.mean(losses)))


def create_accuracy(logits, outputs):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(outputs, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return accuracy


def parser(argv):
    parser = argparse.ArgumentParser(description='Trains a Deep Neural Network on Fashion MNIST Data')
    parser.add_argument('--train_csv', default='training.csv', type=str, required=True, help='Path to the training csv.')
    parser.add_argument('--validation_csv', default='validation.csv', type=str, help='Path to the validation csv.')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch Size of one iteration.')
    parser.add_argument('--buffer_size', default=10000, type=int, help='Buffer Size for random selection of images.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate.')
    parser.add_argument('--nrof_epochs', default=20, type=int, help='Number of Epochs for training.')
    parser.add_argument('--log_dir', default='./log', type=str, help='Location of log.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parser(sys.argv[1:]))