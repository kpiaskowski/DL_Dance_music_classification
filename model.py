import tensorflow as tf


def count_examples(filenames):
    c = 0
    for fn in filenames:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c

def compute_num_batches(filenames, batch_size):
    c = count_examples(filenames)
    return c // batch_size + 1


def decode_data(batch_size, filenames, feature_name, img_height, img_width):
    feature = {feature_name + '/image': tf.FixedLenFeature([], tf.string),
               feature_name + '/label': tf.FixedLenFeature([], tf.int64)}
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.reshape(tf.decode_raw(features[feature_name + '/image'], tf.float32), [img_height, img_width])
    label = tf.cast(features[feature_name + '/label'], tf.int32)

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            capacity=10000,
                                            num_threads=1,
                                            min_after_dequeue=5000,
                                            allow_smaller_final_batch=True)
    return images, labels


def define_model(images_placeholder, labels_placeholder, dropout_placeholder, eta, num_classes):
    inputs = tf.expand_dims(images_placeholder, 3)

    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(inputs,
                                 filters=64,
                                 kernel_size=(2, 2),
                                 strides=(2, 2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.elu)
        max_pool1 = tf.layers.max_pooling2d(conv1, (2, 2), 1, padding='same')

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(max_pool1,
                                 filters=128,
                                 kernel_size=(2, 2),
                                 strides=(2, 2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.elu)
        max_pool2 = tf.layers.max_pooling2d(conv2, (2, 2), 1, padding='same')

    with tf.variable_scope('conv3'):
        conv3 = tf.layers.conv2d(max_pool2,
                                 filters=256,
                                 kernel_size=(2, 2),
                                 strides=(2, 2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.elu)
        max_pool3 = tf.layers.max_pooling2d(conv3, (2, 2), 1, padding='same')

    with tf.variable_scope('conv4'):
        conv4 = tf.layers.conv2d(max_pool3,
                                 filters=512,
                                 kernel_size=(2, 2),
                                 strides=(2, 2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.elu)
        max_pool4 = tf.layers.max_pooling2d(conv4, (2, 2), 1, padding='same')

    with tf.variable_scope('flattening_convolution'):
        conv5 = tf.layers.conv2d(max_pool4,
                                 filters=1,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.elu)

    with tf.variable_scope('fully_connected'):
        flatten = tf.layers.flatten(conv5)
        dense = tf.layers.dense(flatten, 1024, activation=tf.nn.elu)
        dropout = tf.layers.dropout(dense, rate=0.5, training=dropout_placeholder)
        logits = tf.layers.dense(dropout, num_classes)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_placeholder, logits=logits)
    train_op = tf.train.AdamOptimizer(eta).minimize(loss)
    softmax_logits = tf.nn.softmax(logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_placeholder, tf.argmax(softmax_logits, 1, output_type=tf.int32)), dtype=tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    return train_op, loss, accuracy, softmax_logits, merged

