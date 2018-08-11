import numpy as np
import requests as rq
import shutil
import gzip
from matplotlib.pyplot import imshow, show


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
