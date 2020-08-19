import numpy as np

class LogLoss:
	@staticmethod
	def calculate_loss(x, y):
		return -np.log(x[y])
	
	@staticmethod
	def derivative(x, y):
		grad = np.zeros(len(x))
		grad[y] = -1 / x[y]
		return grad
