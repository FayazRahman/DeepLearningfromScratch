import numpy as np

class Relu:
	@staticmethod
	def fn(x):
		return np.maximum(0, x)
	
	@staticmethod
	def derivative(x):
		return (x > 0) * 1
