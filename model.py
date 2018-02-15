import tensorflow as tf

def graph(x, training_mode=True):
    
    with tf.name_scope('Reshape'):
        x = tf.reshape(x, [-1, 28, 28, 1])
    
    print('INPUT SHAPE: ', x.get_shape())
    
    with tf.name_scope('Convolution'):
        with tf.variable_scope('conv2D_A'):
            x = tf.layers.conv2d(x, filters=64,
                                 kernel_size=5, strides=1,
                                 padding='SAME', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                                                                                                seed=100,
                                                                                                dtype=tf.float32),
                                activation=None, 
                                name='conv2d_A')

    print('CONV_A SHAPE: ', x.get_shape())
            
    with tf.name_scope('Batch_Normalization'):
        with tf.variable_scope('batch_norm_A'):
            x = tf.nn.relu6(tf.layers.batch_normalization(x, training=training_mode))
             
    print('BATCH_NORM_A SHAPE: ', x.get_shape())
          
    with tf.name_scope('Avg_Pooling'):
        with tf.variable_scope('avg_pooling_A'):
            x = tf.layers.average_pooling2d(x, pool_size=4, strides=2, padding='same')
            
    print('AVG_POOLING_A SHAPE: ', x.get_shape())
    
    with tf.name_scope('Convolution'):
        with tf.variable_scope('conv2D_B'):
            x = tf.layers.conv2d(x, filters=128,
                                 kernel_size=5, strides=1,
                                 padding='SAME', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                                                                                                seed=100,
                                                                                                dtype=tf.float32),
                                activation=None, 
                                name='conv2d_A')

    print('CONV_B SHAPE: ', x.get_shape())
            
    with tf.name_scope('Batch_Normalization'):
        with tf.variable_scope('batch_norm_B'):
            x = tf.nn.relu6(tf.layers.batch_normalization(x, training=training_mode))
             
    print('BATCH_NORM_B SHAPE: ', x.get_shape())
          
    with tf.name_scope('Avg_Pooling'):
        with tf.variable_scope('avg_pooling_B'):
            x = tf.layers.average_pooling2d(x, pool_size=4, strides=2, padding='same')
            
    print('AVG_POOLING_B SHAPE: ', x.get_shape())
    
    x = tf.reshape(x, (-1, 7*7*128))
    
    print('FC1 INPUT SHAPE: ', x.get_shape())
    
    with tf.name_scope('Fc1'):
        with tf.variable_scope('fully_connc_A'):
            x = tf.layers.dense(x, units=1024, activation=tf.nn.relu6)
            
    print('SOFTMAX INPUT SHAPE: ', x.get_shape())
    
    with tf.name_scope('Softmax'):
        with tf.variable_scope('softmax_A'):
            logits = tf.layers.dense(x, units=10)
    
    print('FINAL OUTPUT SHAPE: ', logits.get_shape())
    
    return logits