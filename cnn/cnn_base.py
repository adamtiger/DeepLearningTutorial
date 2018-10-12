import numpy as np


class Convolution:

    def __init__(self, input_shape, filters, kernel_size, strides, dilations, padding, activation):
        '''
        Reference implementation of a 2-dimensional Convolution operation. 
        Performance has not priority here.
        input_shape - tuple or list with the shape (batch, height, width, channel)
        filters - number of filters to use (this will be the number of channels in the output)
        kernel_size - the size of the kernel window, tuple or list with two elements
        strides - the stride (the window slides to a next position) for the kernels, tuple or list with two elements
        dilations - tuple or list
        padding - string, can be VALID or SAME
        activation - applied after the convolution is done
        '''
        self.batch, self.height, self.width, self.channel = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        self.padding = padding
        self.activation = activation
        
        # initialize the kernel with random numbers to break the symmetry
        self.kernel = np.random.rand(filters, kernel_size[0], kernel_size[1], self.channel)

        # calculate the output shape
        self.output_size = 0
        if self.padding == 'SAME':
            self.output_size = (self.batch, self.height, self.width, self.filters)
        elif self.padding == 'VALID':
            window_size = [(self.kernel_size[0] - 1) * (self.dilations[0] + 1) + 1, (self.kernel_size[1] - 1) * (self.dilations[1] + 1) + 1]
            output_height = (self.height - window_size[0]) // self.strides[0] + 1
            output_width = (self.width - window_size[1]) // self.strides[1] + 1
            self.output_size = (self.batch, output_height, output_width, self.channel)
        else:
            raise AttributeError('Unknown type of padding!')

    def output_shape(self):
        return self.output_size

    def execute(self, x):
        if self.padding == 'SAME':
            window_size = [(self.kernel_size[0] - 1) * (self.dilations[0] + 1) + 1, (self.kernel_size[1] - 1) * (self.dilations[1] + 1) + 1]
            pad_x = ((self.output_size[0] - 1) * self.strides[0] - (self.height - window_size[0])) // 2
            pad_y = ((self.output_size[1] - 1) * self.strides[1] - (self.height - window_size[1])) // 2

            temp = np.zeros((x.shape[0], x.shape[1] + 2 * pad_x, x.shape[2] + 2 * pad_y, x.shape[3]))
            temp[:, pad_x:(x.shape[1] + pad_x), pad_y:(x.shape[2] + pad_y), :] = x[:, :, :, :]
            x = temp

        y = np.zeros(shape=self.output_size)
        for b in range(y.shape[0]):
            for h in range(y.shape[1]):
                for w in range(y.shape[2]):
                    for c in range(y.shape[3]):

                        for k_h in range(self.kernel.shape[1]):
                            for k_w in range(self.kernel.shape[2]):
                                i_h = k_h + h * self.strides[0]
                                i_w = k_w + w * self.strides[1]
                                for k_c in range(self.kernel.shape[3]):
                                    y[b, h, w, c] += self.kernel[c, k_h, k_w, k_c] * x[b, i_h, i_w, k_c]
        return y

    def __call__(self, x):
        self.execute(x)
