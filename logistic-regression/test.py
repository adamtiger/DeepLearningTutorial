import mnist

def test_mnist_read_imgs():
    
    link = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    saving_name = "data/training_mnist_imgs.gz"
    mnist.download(link, saving_name)
    fname = mnist.unzip(saving_name)
    #fname = "data/training_mnist_imgs.mnist"
    _, _, rows, cols, images = mnist.read_img(fname)
    mnist.show_handwritten_digit(images[0], rows, cols)

test_mnist_read_imgs()


def test_mnist_read_labels():
    
    link = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    saving_name = "data/training_mnist_lbs.gz"
    mnist.download(link, saving_name)
    fname = mnist.unzip(saving_name)
    #fname = "data/training_mnist_lbs.mnist"
    _, _, labels = mnist.read_label(fname)
    print(labels[0])

test_mnist_read_labels()