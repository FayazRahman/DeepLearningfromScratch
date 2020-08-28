import numpy as np


class Dense:
    def __init__(self, input_len, nodes, activation):
        self.input_len = input_len
        self.nodes = nodes
        self.activation = activation
        self.W = np.random.randn(input_len, nodes) / np.sqrt(input_len)
        self.b = np.random.randn(nodes)

    def __repr__(self):
        return "Dense ({}, {})".format(self.input_len, self.nodes)

    def forward(self, inpt):
        self.inpt_shape = inpt.shape
        inpt = inpt.flatten()
        self.inpt = inpt
        assert inpt.shape[0] == self.input_len, "Input length does not match"

        output = self.activation.fn(inpt.dot(self.W) + self.b)
        self.out = output

        return output

    def backprop(self, dl_dout, alpha):
        out = self.out

        dl_dt = dl_dout * self.activation.derivative(out)

        dt_dinputs = self.W.T
        dt_dw = self.inpt[np.newaxis].T
        dt_db = 1

        dl_dw = dt_dw.dot(dl_dt[np.newaxis])
        dl_db = dl_dt * dt_db
        dl_dinputs = dl_dt.dot(dt_dinputs)

        self.W += -alpha * dl_dw
        self.b += -alpha * dl_db

        return dl_dinputs.reshape(self.inpt_shape)
