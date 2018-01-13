import os
import time
import tensorflow as tf

from model import decode_data, define_model, compute_num_batches

dictionary = sorted(os.listdir('downloaded_data'))

# params
mode = 'train'
model_name = 'model1'
img_width = 512
img_height = 128
eta=0.001
num_epochs = 200
train_batch = 100
val_test_batch = 50
num_classes = len(dictionary)

train_filenames = [os.path.join('preprocessed_data', filename) for filename in os.listdir('preprocessed_data') if 'train' in os.path.join('preprocessed_data', filename)]
validation_filenames = [os.path.join('preprocessed_data', filename) for filename in os.listdir('preprocessed_data') if 'validation' in os.path.join('preprocessed_data', filename)]
test_filenames = [os.path.join('preprocessed_data', filename) for filename in os.listdir('preprocessed_data') if 'test' in os.path.join('preprocessed_data', filename)]

train_num_batches = compute_num_batches(train_filenames, train_batch)
validation_num_batches = compute_num_batches(validation_filenames, val_test_batch)
test_num_batches = compute_num_batches(test_filenames, val_test_batch)

if mode=='train':
    with tf.Session() as sess:

        # train and validation data
        train_images, train_labels = decode_data(train_batch, train_filenames, 'train', img_height=img_height, img_width=img_width)
        validation_images, validation_labels = decode_data(val_test_batch, validation_filenames, 'validation', img_height=img_height, img_width=img_width)

        images_placeholder = tf.placeholder(tf.float32, shape=[None, img_height, img_width])
        labels_placeholder = tf.placeholder(tf.int32, shape=None)
        dropout_placeholder = tf.placeholder(tf.bool, shape=())

        # ops
        train_op, loss, accuracy, softmax_logits, merged = define_model(images_placeholder, labels_placeholder, dropout_placeholder, eta, num_classes)

        # visualization and saving
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('summaries/' + model_name + '/train', sess.graph)
        val_writer = tf.summary.FileWriter('summaries/' + model_name + '/validation')

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if os.path.isdir('models/' + model_name):
            saver.restore(sess, 'models/' + model_name + '/model.ckpt')
            print('Model read')

        # coordinating data readers
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # training
        for epoch in range(num_epochs):
            for batch_index in range(train_num_batches):
                img, lbl = sess.run([train_images, train_labels])
                sess.run(train_op, feed_dict={images_placeholder: img, labels_placeholder: lbl, dropout_placeholder: False})
                print('\rEpoch: %d of %d, batch_n: %d of %d' % (epoch+1, num_epochs, batch_index+1, train_num_batches), end='', flush=True)

            # training measures and visualisation - just for showing progress
            img, lbl = sess.run([train_images, train_labels])
            summ, train_loss, train_acc = sess.run([merged, loss, accuracy], feed_dict={images_placeholder: img, labels_placeholder: lbl, dropout_placeholder: False})
            train_writer.add_summary(summ, epoch)
            train_writer.flush()

            # validation measures and visualisation - just for showing progress
            img, lbl = sess.run([validation_images, validation_labels])
            summ, validation_loss, validation_acc = sess.run([merged, loss, accuracy], feed_dict={images_placeholder: img, labels_placeholder: lbl, dropout_placeholder: False})
            val_writer.add_summary(summ, epoch)
            val_writer.flush()

            # textual description of measures
            print('\nTraining loss: %.4f, training acc: %.3f' % (train_loss, train_acc))
            print('Validation loss: %.4f, validation acc: %.3f\n' % (validation_loss, validation_acc))

            # saving
            saver.save(sess, 'models/' + model_name + '/model.ckpt')

        # stopping all coordinators etc
        coord.request_stop()
        coord.join(threads)
        sess.close()

elif mode == 'test':
    with tf.Session() as sess:

        # train and validation data
        test_images, test_labels = decode_data(val_test_batch, test_filenames, 'test', img_height=img_height,
                                                 img_width=img_width)

        images_placeholder = tf.placeholder(tf.float32, shape=[None, img_height, img_width])
        labels_placeholder = tf.placeholder(tf.int32, shape=[None])
        dropout_placeholder = tf.placeholder(tf.bool, shape=())

        # ops
        _, _, accuracy, _, _= define_model(images_placeholder,
                                           labels_placeholder,
                                           dropout_placeholder,
                                           eta,
                                           num_classes)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # visualization and saving
        saver = tf.train.Saver()

        if os.path.isdir('models/' + model_name):
            saver.restore(sess, 'models/' + model_name + '/model.ckpt')
            print('Model read')
        else:
            raise Exception("Model not found! Check if model_name is correct!")

        # coordinating data readers
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # test
        accuracies = []
        for batch_index in range(val_test_batch):
            img, lbl = sess.run([test_images, test_labels])
            acc = sess.run(accuracy, feed_dict={images_placeholder: img, labels_placeholder: lbl, dropout_placeholder: False})
            accuracies.append(acc)
        print('Total accuracy: %f' % (sum(accuracies) / len(accuracies)))

        # stopping all coordinators etc
        coord.request_stop()
        coord.join(threads)
        sess.close()

else:
    raise Exception("mode must be either 'train' or 'test'!")

