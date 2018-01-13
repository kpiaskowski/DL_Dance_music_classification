import hashlib
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from utils import extract_spectrogram_windows
import tensorflow as tf

try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve


class DataProvider:
    def __init__(self, download_folder='downloaded_data/', preprocessed_data_folder='preprocessed_data/',
                 xml_folder='./'):
        self.download_folder = download_folder
        self.xml_folder = xml_folder
        self.preprocessed_data_folder = preprocessed_data_folder
        self.train_ratio = 0.8
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.download_data()
        print('Preprocessing data')
        self.preprocess_data()

    def download_data(self):
        threading = True
        n_thread = 8
        skip_already_downloaded_mp3 = True
        version = '1.1'
        mp3_link_template = 'http://media.ballroomdancers.com/mp3/{}.mp3'

        def open_xml(folder, version):
            filename = os.path.join(folder, 'extendedballroom_v{}.xml'.format(version))
            tree = ET.parse(filename).getroot()
            return tree

        def hashfile(afile, hasher, blocksize=65536):
            buf = afile.read(blocksize)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(blocksize)
            return hasher.hexdigest()

        def download(list_, threading=True, n_thread=8):

            print('Downloading {} files. This may take a while.'.format(len(list_)))
            if threading:
                try:
                    import queue
                except:
                    import Queue as queue
                import threading
                import time

                exit_flag = False

                class ThreadDLMp3(threading.Thread):
                    """Thread to download multiple files at once"""

                    def __init__(self, queue):
                        threading.Thread.__init__(self)
                        self.queue = queue

                    def run(self):
                        while not exit_flag:
                            mp3_link, mp3_filepath = self.queue.get()
                            if os.path.exists(mp3_filepath):
                                os.remove(mp3_filepath)
                            urlretrieve(mp3_link, mp3_filepath)
                            self.queue.task_done()

                q = queue.Queue()
                threads = []
                for i in range(n_thread):
                    t = ThreadDLMp3(q)
                    t.setDaemon(True)
                    t.start()
                    threads.append(t)

            for mp3_link, mp3_filepath in list_:
                if threading:
                    q.put([mp3_link, mp3_filepath])
                else:
                    urlretrieve(mp3_link, mp3_filepath)

            if threading:
                try:
                    ##Wait for queue to be exhausted and then exit main program
                    while not q.empty():
                        pass
                    # Stop the threads
                    exit_flag = True
                except (KeyboardInterrupt, SystemExit) as e:
                    sys.exit(e)

        def check(folder, xml_root, sublist=[]):
            filelist = []
            hashlist = []
            download_again = []
            print('Checking dataset')
            for genre_node in xml_root:
                genre_folder = os.path.join(self.download_folder, genre_node.tag)
                for song_node in genre_node:
                    song_id = song_node.get('id')
                    mp3_filepath = os.path.join(genre_folder, song_id + '.mp3')
                    if sublist and mp3_filepath not in sublist:
                        continue
                    h = hashfile(open(mp3_filepath, 'rb'), hashlib.md5())
                    status = song_node.get('hash') == h
                    filelist.append(mp3_filepath)
                    hashlist.append(status)

                    if not status:
                        print('Error with file {}, expected hash {}, found hash {}'
                              .format(mp3_filepath, song_node.get('hash'), h))
                        download_again.append((mp3_link_template.format(song_id), mp3_filepath))

            print('{} out of {} are valid'.format(sum(hashlist), len(filelist)))

            return download_again

        xml_root = open_xml(self.xml_folder, version)

        download_list = []

        if not os.path.exists(self.download_folder):
            os.mkdir(self.download_folder)
        for genre_node in xml_root:
            genre_folder = os.path.join(self.download_folder, genre_node.tag)
            if not os.path.exists(genre_folder):
                os.mkdir(genre_folder)
            for song_node in genre_node:
                song_id = song_node.get('id')
                mp3_filepath = os.path.join(genre_folder, song_id + '.mp3')
                if skip_already_downloaded_mp3 and os.path.exists(mp3_filepath):
                    continue
                download_list.append((mp3_link_template.format(song_id), mp3_filepath))

        download(download_list, threading, n_thread)

        download_again = check(self.download_folder, xml_root)
        while download_again:
            download(download_again, threading, n_thread)
            time.sleep(2)
            download_again = check(self.download_folder, xml_root, sublist=[b for a, b in download_again])

    def preprocess_data(self):
        genres = os.listdir(self.download_folder)
        dictionary = sorted(genres)
        if not os.path.exists(self.preprocessed_data_folder):
            os.mkdir(self.preprocessed_data_folder)
        else:
            saved_genres = [g.replace('_train.tfrecord', '') for g in os.listdir(self.preprocessed_data_folder)]
            genres = set(genres) - set(saved_genres)
        for k, genre in enumerate(genres):
            train_writer, validation_writer, test_writer = self.create_writers(genre)
            files = os.listdir(os.path.join(self.download_folder, genre))
            random.shuffle(files)
            max_train_id = int(self.train_ratio * len(files))
            max_validation_id = int((self.train_ratio + self.validation_ratio) * len(files))
            for i, file in enumerate(files):
                print("\rPreprocessing data: genre %d of %d, file %d of %d" % (k + 1, len(genres), i + 1, len(files)),
                      end='')
                filepath = os.path.join(self.download_folder, genre, file)
                spectrogram_windows = extract_spectrogram_windows(filepath)
                for window in spectrogram_windows:
                    if i < max_train_id:
                        feature = {'train/label': self._int64_feature(dictionary.index(genre)),
                                   'train/image': self._bytes_feature(tf.compat.as_bytes(window.tostring()))}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        train_writer.write(example.SerializeToString())
                    elif i < max_validation_id:
                        feature = {'validation/label': self._int64_feature(dictionary.index(genre)),
                                   'validation/image': self._bytes_feature(tf.compat.as_bytes(window.tostring()))}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        validation_writer.write(example.SerializeToString())
                    else:
                        feature = {'test/label': self._int64_feature(dictionary.index(genre)),
                                   'test/image': self._bytes_feature(tf.compat.as_bytes(window.tostring()))}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        test_writer.write(example.SerializeToString())
            train_writer.close()
            validation_writer.close()
            test_writer.close()

    def create_writers(self, genre):
        train_writer = tf.python_io.TFRecordWriter(
            os.path.join(self.preprocessed_data_folder, genre + '_train.tfrecord'))
        validation_writer = tf.python_io.TFRecordWriter(
            os.path.join(self.preprocessed_data_folder, genre + '_validation.tfrecord'))
        test_writer = tf.python_io.TFRecordWriter(os.path.join(self.preprocessed_data_folder, genre + '_test.tfrecord'))
        return train_writer, validation_writer, test_writer

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


provider = DataProvider()

