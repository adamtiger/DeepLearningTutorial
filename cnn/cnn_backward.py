import numpy as np


class ConvGrad:

    def __init__(self, cnn_layer):
        '''
        The calculation of the gradient of convolution
        in order to backprop the errors.
        '''
        self.cnn_layer = cnn_layer

    def gradient(self, x):
        o_b, o_h, o_w, o_f = self.cnn_layer.output_shape()
        (k_h, k_w), k_c = self.cnn_layer.kernel_size, self.cnn_layer.channel  # kernel's filter number is the same as the channel in the output
        J = np.zeros((o_b, o_h, o_w, o_f, k_h, k_w, k_c))

        for b in range(J.shape[0]):
            for h in range(J.shape[1]):
                for w in range(J.shape[2]):

                    for i in range(J.shape[4]):
                        for j in range(J.shape[5]):
                            i_h = i + h * self.cnn_layer.strides[0]
                            i_w = j + w * self.cnn_layer.strides[1]
                            for c in range(J.shape[6]):
                                J[b, h, w, :, i, j, c] = x[b, i_h + i, i_w + j, c]
        return J

    def backrop(self, delta, x):
        k_f, (k_h, k_w), k_c = self.cnn_layer.filters, self.cnn_layer.kernel_size, self.cnn_layer.channel  # kernel's filter number is the same as the channel in the output
        J = self.gradient(x)
        d_w = np.zeros((k_f, k_h, k_w, k_c))
        
        for b in range(J.shape[0]):
            for h in range(J.shape[1]):
                for w in range(J.shape[2]):
                    for f in range(J.shape[3]):
                        for i in range(J.shape[4]):
                            for j in range(J.shape[5]):
                                for c in range(J.shape[6]):
                                    d_w[f, i, j, c] += delta[b, h, w, f] * J[b, h, w, f, i, j, c]
        return d_w
