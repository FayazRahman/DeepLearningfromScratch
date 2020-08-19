import numpy as np

class Model:
	def __init__(self, loss_fn, *args):
		self.loss_fn = loss_fn
		self.layers = args
	
	def __repr__(self):
		return ' => '.join([str(layer) for layer in self.layers])
	
	def fit(self, X, Y, epochs, alpha):
		for epoch in np.arange(epochs):
			curr_loss = 0
			curr_acc = 0
			for i, (x, y) in enumerate(zip(X, Y)):
				output = x
				for layer in self.layers:
					output = layer.forward(output)
				
				curr_loss += self.loss_fn.calculate_loss(output, y)
				curr_acc += 1 if np.argmax(output) == y else 0

				grad = self.loss_fn.derivative(output, y)
				for layer in (self.layers[::-1]):
					grad = layer.backprop(grad, alpha)

				if (i + 1) % 100 == 0:
					print("[INFO] [Step: {}] Loss: {}, Accuracy: {}".format(i + 1, curr_loss / 100, curr_acc))
					curr_loss, curr_acc = 0, 0

	def predict(self, X):
		n_samples = X.shape[0]
		output = []
		for i in range(n_samples):
			out = X[i]
			for layer in self.layers:
				out = layer.forward(out)
			output.append(out)
		return output
	
	def validate(self, X, Y):
		print("Testing CNN...")
		output = self.predict(X)
		loss = 0
		acc = 0
		for out, y in zip(output, Y):
			loss += self.loss_fn.calculate_loss(out, y)
			acc += 1 if np.argmax(out) == y else 0
		print("[INFO] Test Loss: {}".format(loss / len(X)))
		print("[INFO] Test Accuracy: {}".format(acc / len(X)))
		return loss, acc
