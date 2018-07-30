import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import glob
import os
from keras.utils import to_categorical

def download_images():
    """Loads images with urls from url_files/{class}.txt and stores them in data/{class}/"""

    url_files = glob.glob('url_files/*.txt')
    for url_file_path in url_files:
        label = re.split(r"/|\.", url_file_path)[-2]
        datadir = 'data/'+label+'/'
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        with open(url_file_path) as url_file:
            urlset = set(url_file.readlines())
            setlen = float(len(urlset))
            for i, url in enumerate(urlset):
                if i % 10 ==0:
                    print('progress {}: {}%'.format(label, int(100 * float(i)/setlen)))
                filename = re.sub(r'\W+', '', url)
                savepath = datadir+filename+'.jpg'
                if os.path.isfile(savepath):
                    print('already loaded')
                    continue
                try:
                    response = requests.get(url, timeout=5)
                    if response.history: #removed flickr images redirect to an empty image
                        continue
                    img = Image.open(BytesIO(response.content))
                    if not np.array(img).any() or np.array(img).shape[2] != 3:
                        raise('Downloaded image was empty or not 3 channels')
                    img.save(savepath)
                except Exception as e:
                    print('Error for {}: '.format(url), e.args)
                    continue


def get_n_random(x,n):
    shuffled_indexes = np.random.permutation(x.shape[0])
    x = np.take(x, shuffled_indexes, axis=0, out=x)
    return x[:n]


def shuffle_dataset(x, y):
    shuffled_indexes = np.random.permutation(x.shape[0])
    x = np.take(x, shuffled_indexes, axis=0, out=x)
    y = np.take(y, shuffled_indexes, axis=0, out=y)
    return x,y


def load_data(n_perLabel, target_dim):
    """returns (trainx, trainy), (testx, testy). images loaded and labeled from {label} in /data/{label}/"""
    train_fraction = 0.8
    data_dir = glob.glob('data/*')
    n_labels = len(data_dir)
    data_x = np.ndarray((sum(n_perLabel),target_dim[0],target_dim[1],3))
    data_y = np.ndarray((sum(n_perLabel),n_labels))

    for label_num, dir in enumerate(data_dir):
        label = dir.split('/')[-1]
        files = np.array(glob.glob(dir+'/*.jpg'))
        n_files = len(files)
        n_filesToLoad = n_perLabel[label_num]
        if n_files < n_filesToLoad:
            raise Exception('Tried to load {} samples from {}. Insufficient samples!'.format(n_filesToLoad, dir))
        random_files = get_n_random(files, n_filesToLoad)
        for file_num, file in enumerate(random_files):
            try:
                if file_num%100 == 0:
                    print('progress loading {}: {}%'.format(label, 100*file_num/n_filesToLoad))
                img = np.array(Image.open(file).resize(target_dim))
                data_x[sum(n_perLabel[:label_num]) + file_num, :, :] = img
                data_y[sum(n_perLabel[:label_num]) + file_num, :] = to_categorical(label_num, n_labels)
            except Exception as e:
                print('Error: {}'.format(file))
                raise(e)

    data_x, data_y = shuffle_dataset(data_x, data_y)

    split_index = int(train_fraction*len(data_x))
    train_x = data_x[:split_index]
    train_y = data_y[:split_index]
    test_x = data_x[split_index:]
    test_y = data_y[split_index:]

    return (train_x, train_y), (test_x, test_y)




