import numpy as np

class Sigmoid:
	@staticmethod
	def fn(x):
		return 1. / (1 + np.exp(-x))
	
	@staticmethod
	def derivative(x):
		return x * (1 - x)
