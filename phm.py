import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import tensorflow.keras as keras


class PHMDense(keras.layers.Layer):
    def __init__(self, n, units, activation="swish"):
        super(PHMDense, self).__init__()
        self.n = n
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        n = self.n
        k = self.units
        d = input_shape[-1]
        self.h = k
        self.w = d
        self.kernel_a = self.add_weight(shape=(1, n, n, 1))
        self.kernel_s = self.add_weight(shape=(k//n, d//n, 1, 1))
        self.bias = self.add_weight(shape=(k,))

    def call(self, inputs, *args, **kwargs):
        kernel = tf.squeeze(tf.nn.conv2d_transpose(self.kernel_a, self.kernel_s, (1, self.h, self.w, 1), (1, self.h//self.n, self.w//self.n, 1), "VALID"))
        o = tf.matmul(inputs, kernel, transpose_b=True)
        o = tf.nn.bias_add(o, self.bias)
        o = self.activation(o)
        return o


if __name__ == "__main__":
    dataset = make_blobs(n_samples=500, n_features=2, centers=2)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)


