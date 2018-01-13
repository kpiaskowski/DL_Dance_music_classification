import tensorflow as tf
import os
from model import define_model
from utils import extract_spectrogram_windows
from scipy.stats import mode

dictionary = sorted(os.listdir('downloaded_data'))
model_name = 'model1'
num_classes = len(dictionary)
img_width = 512
img_height = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    images_placeholder = tf.placeholder(tf.float32, shape=[None, img_height, img_width])
    labels_placeholder = tf.placeholder(tf.int32, shape=None)
    dropout_placeholder = tf.placeholder(tf.bool, shape=())

    _, _, _, softmax_logits, _ = define_model(images_placeholder, labels_placeholder, dropout_placeholder, eta=0.001, num_classes=num_classes)
    classes = tf.argmax(softmax_logits, axis=1)

    saver = tf.train.Saver()
    if os.path.isdir('models/' + model_name):
        saver.restore(sess, 'models/' + model_name + '/model.ckpt')
        print('Model read')
    else:
        raise Exception("Model not found! Check if model_name is correct!")

    while True:
        filename = input("Type filename (or 'q' to quit): ")
        path = os.path.join('sample_audio', filename)
        if filename == 'q':
            break
        elif os.path.isfile(path):
            windows = extract_spectrogram_windows(path)
            class_predictions = sess.run(classes, feed_dict={images_placeholder: windows, dropout_placeholder: False})
            top_class = mode(class_predictions)[0][0]
            print(dictionary[top_class])
        else:
            print("File doesn't exist!")
            continue

