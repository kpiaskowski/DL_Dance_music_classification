# Dance music classifier
Repository contains TensorFlow model of convolutional network, designed to classify 7 genres of classical dance music (chacha, rumba, foxtrot, quickstep, sambo, tango and waltz).
The data comes from modified Extended Ballroom dataset. I modified downloading script in order to download only seven largest genres (out of initial 13).

## How does it work?
This work is based on great article by Julien Despois, available here: https://chatbotslife.com/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194.
In order to classify music, we need to do some preprocessing. Originally, all tracks comes in MP3 format. Every track is just a subset of time samples:

![Sample chacha timeseries](readme_images/timeseries.png?raw=true "Sample chacha timeseries")

To utilize convolutional network, we need to convert such data to spectrograms. Example of spectrogram:

![Sample chacha spectrogram](readme_images/spectrogram.png?raw=true "Sample chacha spectrogram")

Single spectrogram is too long for our network. I cut spectrogram of every track into many 128x512px windows. Each window occupies about 2.6 seconds of music.
That's pretty much of it. Each window is labelled and feed into convolutional network. The architecture is rather simple: 4 convolution layers, each followed by max pooling. 
I added one dense layer and one output layer after convolutions. 
If you want to know more about technical issues of spectrograms and timeseries, I strongly recommend to read aforementioned article.

## How to run it?
1. Run **data_provider.py**. It will download all needed data and convert it to TFRecord format. Both downloading and conversion may take some time, because there are about 4000 tracks (about 3GB in total). 
2. I attached some pretrained model. In order to try it, just run **infer.py** and type some filenames in console when asked to. Your data should be in *sample_audio* and you should type only filename along with extension, for eg. 'chacha.mp3'.
3. If you want to train your own model, run **trainer.py**. You might want to change *model_name* param in order not to overwrite pretrained model.

## Results
The network achieves a decent 88% accuracy on test set. It was trained for 40 epochs on GeForce 1080Ti. There is still a lot of room for improvements - the network recognizes *Despacito* as samba - I checked! :D

## Dependencies
I worked in Python 3.6.3 with following libraries installed:
- TensorFlow 1.4.0
- numpy 1.13.3
- scipy 1.0.0
- scikit-learn 0.19.1
- opencv3 3.1.0
but you should be able to run it on lower versions of these libraries.

## Contact
In case of any questions, feel free to contact me at kar.piaskowski@gmail.com. 
