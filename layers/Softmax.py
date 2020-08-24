import numpy as np

class Softmax:
	def __init__(self, input_len, nodes):
		self.input_len = input_len
		self.nodes = nodes
		self.W = np.random.randn(input_len, nodes) / np.sqrt(input_len)
		self.b = np.zeros(nodes)

	def __repr__(self):
		return "Softmax ({}, {})".format(self.input_len, self.nodes)

	def forward(self, inpt):
		self.inpt_shape = inpt.shape
		inpt = inpt.flatten()
		assert inpt.shape[0] == self.input_len, "Input length does not match"

		self.inpt = inpt
		output = np.exp(inpt.dot(self.W) + self.b)

		self.t_exp = output
		self.S = np.sum(output, axis=0)

		return output / self.S
	
	def backprop(self, dl_dout, alpha):
		for i, grad in enumerate(dl_dout):
			if grad == 0:
				continue
			
			t_exp = self.t_exp
			S = self.S

			dout_dt = -t_exp[i] * t_exp / (S ** 2) 
			dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

			dt_dw = self.inpt[np.newaxis].T
			dt_db = 1

			dl_dt = grad * dout_dt

			dl_dw = dt_dw.dot(dl_dt[np.newaxis])
			dl_db = dl_dt * dt_db
			dl_dinputs = dl_dt.dot(self.W.T)

			self.W += -alpha * dl_dw
			self.b += -alpha * dl_db

			return dl_dinputs.reshape(self.inpt_shape)