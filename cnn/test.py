from cnn import cnn_base


def test_cnn_base():

    conv = cnn_base.Convolution((32, 128, 128, 3), 12, (3, 3), (1, 1), (0, 0), 'VALID', lambda x: x)
    x = 0
    conv(x)
    print("OK: test_cnn_base")

test_cnn_base()
