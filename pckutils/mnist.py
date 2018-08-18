import numpy as np
import requests as rq
import shutil
import gzip
from matplotlib.pyplot import imshow, show
import os


def download(link, saved_name):
    downloaded = rq.get(link, allow_redirects=True)
    with open(saved_name, 'wb') as f:
        f.write(downloaded.content)
    

def unzip(file_name):
    new_file_name = file_name.replace('.gz', '.mnist')
    with gzip.open(file_name, 'rb') as src:
        with open(new_file_name, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    return new_file_name


int32 = np.dtype(np.int32).newbyteorder('>')
def read_int32(bin_file):
        int32 = np.dtype(np.int32).newbyteorder('>')
        return np.frombuffer(bin_file.read(4), dtype=int32)[0]

uint8 = np.dtype(np.uint8).newbyteorder('>')
def read_uint8(bin_file):
        return np.frombuffer(bin_file.read(1), dtype=uint8)[0]


def read_img(file_name):
    
    # read the file parts
    images = []

    with open(file_name, 'rb') as imgs:

        # read magic number, number of images, rows and cols of an image
        mgb = read_int32(imgs)
        num_of_imgs = read_int32(imgs)
        rows = read_int32(imgs)
        cols = read_int32(imgs)
        print(mgb, num_of_imgs, rows, cols)

        for i in range(num_of_imgs):
            
            # read the pixels for an image
            IMG = np.zeros(rows*cols, dtype=np.uint8)

            for px in range(rows*cols):
                IMG[px] = read_uint8(imgs)

            images.append(IMG)

            if (i + 1) % (num_of_imgs/20) == 0:
                print("Reading images: [%d%%]\r" %int((i+1)/num_of_imgs * 100), end="")
    print("")

    return mgb, num_of_imgs, rows, cols, images 


def read_label(file_name):
    
    # read the file parts
    labels = []

    with open(file_name, 'rb') as lbs:

        # read magic number, number of images
        mgb = read_int32(lbs)
        num_of_imgs = read_int32(lbs)
        print(mgb, num_of_imgs)

        for i in range(num_of_imgs):

            LABEL = read_uint8(lbs)
            labels.append(LABEL)

            if (i + 1) % (num_of_imgs/20) == 0:
                print("Reading labels: [%d%%]\r" %int((i+1)/num_of_imgs * 100), end="")
    print("")

    return mgb, num_of_imgs, labels


def show_handwritten_digit(img_mtx, rows, cols):
    imshow(img_mtx.reshape((rows, cols)))
    show()


def separate_labels(xs, ys, labels):
    '''
    Chooses the data samples with the given labels.
    xs - list of images in a flat array
    ys - list of corresponding labels
    labels - the labels to gather, e.g. 1,2,3 will be kept but others will be throwen away
    '''
    xs_sep, ys_sep = [], []
    for x, y in zip(xs, ys):
        if y in labels:
            xs_sep.append(x)
            ys_sep.append(y)
    return xs_sep, ys_sep


class MNISTdata:

    def __init__(self, x_train_s, y_train_s, x_test_s, y_test_s, num_train_imgs, num_test_imgs, rows, cols):
        self.X_train = x_train_s
        self.Y_train = y_train_s
        self.X_test = x_test_s
        self.Y_test = y_test_s
        self.num_train_imgs = num_train_imgs
        self.num_test_imgs = num_test_imgs
        self.rows = rows
        self.cols = cols

def load_mnist(folder):
    
    # url addresses for mnist
    url_train_image = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    url_train_label = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    url_test_image = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    url_test_label = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    # download packages if it was not downloaded
    train_imgs_file_name = folder + "/training_mnist_imgs.gz"
    train_lbls_file_name = folder + "/training_mnist_lbls.gz"
    test_imgs_file_name = folder + "/test_mnist_imgs.gz"
    test_lbls_file_name = folder + "/test_mnist_lbls.gz"

    # checking if a file exists
    train_I = os.path.exists(train_imgs_file_name)
    train_L = os.path.exists(train_lbls_file_name)
    test_I = os.path.exists(test_imgs_file_name)
    test_L = os.path.exists(test_lbls_file_name)

    if not train_I:
        download(url_train_image, train_imgs_file_name)
        unzip(train_imgs_file_name)
        print("train_I done.")
    
    if not train_L:
        download(url_train_label, train_lbls_file_name)
        unzip(train_lbls_file_name)
        print("train_L done.")
    
    if not test_I:
        download(url_test_image, test_imgs_file_name)
        unzip(test_imgs_file_name)
        print("test_I done.")
    
    if not test_L:
        download(url_test_label, test_lbls_file_name)
        unzip(test_lbls_file_name)
        print("test_L done.")

    # load in the images and labels
    train_imgs_file_name = "data/training_mnist_imgs.mnist"
    train_lbls_file_name = "data/training_mnist_lbls.mnist"
    test_imgs_file_name = "data/test_mnist_imgs.mnist"
    test_lbls_file_name = "data/test_mnist_lbls.mnist"

    # training images
    mgb, num_train_imgs, rows, cols, x_train_s = read_img(train_imgs_file_name)
    assert mgb == 2051, "Wrong magic number when training images were loaded!"

    # training labels (number of labels are the same as number of images)
    mgb, _, y_train_s = read_label(train_lbls_file_name)
    assert mgb == 2049, "Wrong magic number when training labels were loaded!"

    # test images (test image size is the same)
    mgb, num_test_imgs, _, _, x_test_s = read_img(test_imgs_file_name)
    assert mgb == 2051, "Wrong magic number when test images were loaded!"

    # test labels
    mgb, _, y_test_s = read_label(test_lbls_file_name)
    assert mgb == 2049, "Wrong magic number when test labels were loaded!"
    
    data = MNISTdata(x_train_s, y_train_s, x_test_s, y_test_s, num_train_imgs, num_test_imgs, rows, cols)
    return data

