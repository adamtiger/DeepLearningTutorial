from multilayer_nn import mlp, losses, activations, initializers, optimizers, regularizers
import numpy as np

'''
Example how to use Mlp class.
It also makes possible to debug the code.
'''

# creating data
xs, ys = [], []
samples = 256
input_length = 124
output_length = 5

for _ in range(samples):
    xs.append(np.random.uniform(size=input_length))
    ys.append(np.random.uniform(size=output_length))

# instantiate tools
loss = losses.MeanSquaredError()
initializer = initializers.Xavier()
optimizer = optimizers.SGD(0.01)
regularizer = regularizers.ZeroRegularizer()

# create the network
nn = mlp.Mlp(optimizer, loss, initializer, regularizer)
nn.add_layer(254, input_length, activations.Relu())
nn.add_layer(32, activation=activations.Tanh())
nn.add_layer(output_length, activation=activations.Softmax())

# fitting without callback
nn.fit(xs, ys, 3, 8, verbose=True)

# predicting
print(nn.predict_batch(xs, "onehot"))

print("At least runs.")