import tensorflow as tf

def graph(x, training_mode=True):
    
    with tf.name_scope('Reshape'):
        x = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x, 10)
    
    print('INPUT SHAPE: ', x.get_shape())
    
    with tf.name_scope('Convolution_A'):
        with tf.variable_scope('conv2D_A'):
            x = tf.layers.conv2d(x, filters=32,
                                 kernel_size=[5,5], strides=1,
                                 padding='SAME', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                                                                                                seed=349,
                                                                                                dtype=tf.float32),
                                activation=None, 
                                name='conv2d_A')

    print('CONV_A SHAPE: ', x.get_shape())
    
    with tf.name_scope('Batch_Normalization'):
        with tf.variable_scope('batch_norm_A'):
            x = tf.nn.relu(tf.layers.batch_normalization(x, training=training_mode))
             
    print('BATCH_NORM_A SHAPE: ', x.get_shape())

    with tf.name_scope('Max_Pooling_A'):
        with tf.variable_scope('max_pooling_A'):
            x = tf.layers.max_pooling2d(x, pool_size=[2,2], strides=2, padding='same')
            
    print('MAX_POOLING_A SHAPE: ', x.get_shape())

    with tf.name_scope('Convolution_B'):
        with tf.variable_scope('conv2D_B'):
            x = tf.layers.conv2d(x, filters=64,
                                 kernel_size=[5,5], strides=1,
                                 padding='SAME', 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                                                                                                seed=100,
                                                                                                dtype=tf.float32),
                                activation=tf.nn.relu, 
                                name='conv2d_B')

    print('CONV_B SHAPE: ', x.get_shape())

    with tf.name_scope('Batch_Normalization'):
        with tf.variable_scope('batch_norm_B'):
            x = tf.nn.relu(tf.layers.batch_normalization(x, training=training_mode))
             
    print('BATCH_NORM_B SHAPE: ', x.get_shape())

    with tf.name_scope('Max_Pooling_B'):
        with tf.variable_scope('max_pooling_B'):
            x = tf.layers.max_pooling2d(x, pool_size=[2,2], strides=2, padding='same')
            
    print('MAX_POOLING_B SHAPE: ', x.get_shape())

    x = tf.reshape(x, (-1, 7*7*64))

    # x = tf.reshape(x, [-1, 7*7*32])
    
    print('FC1 INPUT SHAPE: ', x.get_shape())
    
    with tf.name_scope('FC1'):
        with tf.variable_scope('fully_connc_A'):
            x = tf.contrib.layers.fully_connected(x, num_outputs=1024,activation_fn=tf.nn.relu)
            # x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
            # x = tf.layers.dropout(x, rate=0.4, training=True)

    print('FC2 INPUT SHAPE: ', x.get_shape())
    
    # with tf.name_scope('Fc12'):
    #     with tf.variable_scope('fully_connc_B'):
    #         x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
            
    print('SOFTMAX INPUT SHAPE: ', x.get_shape())
    
    tf.summary.histogram('activations', x)

    with tf.name_scope('Softmax'):
        with tf.variable_scope('softmax_A'):
            x = tf.contrib.layers.fully_connected(x, num_outputs=10,activation_fn=None)
            # x = tf.layers.dense(x, units=10)
    
    print('FINAL OUTPUT SHAPE: ', x.get_shape())
    
    return x