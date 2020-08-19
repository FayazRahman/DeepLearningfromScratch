import numpy as np
import mnist

class Conv2D:
	def __init__(self, num_filters, filter_shape, strides=1, padding="same"):
		self.num_filters = num_filters
		self.f_h = filter_shape[0]
		self.f_w = filter_shape[1]
		if len(filter_shape) == 3:
			self.f_n = filter_shape[2]
			self.filters = np.random.randn(self.num_filters, self.f_h, self.f_w, self.f_n) / (self.f_h * self.f_w)
		else:
			self.f_n = 1
			self.filters = np.random.randn(self.num_filters, self.f_h, self.f_w) / (self.f_h * self.f_w)

		self.strides = strides
		self.padding = padding
	
	def __repr__(self):
		return "Conv2D ({}, {}, {})".format(self.num_filters, self.f_h, self.f_w)
	
	def iterate_regions(self, image):
		if image.ndim == 3:
			h, w, n = image.shape
		elif image.ndim == 2:
			h, w = image.shape
			n = 1
		else:
			raise Exception("Expected ndim 2 or 3, received " + str(image.ndim))

		for i in range(0, h - self.f_h + 1, self.strides):
			for j in range(0, w - self.f_w + 1, self.strides):
				im_region = image[i : i + self.f_h, j : j + self.f_w]
				yield i, j, im_region
	
	def forward(self, inpt):
		self.inpt = inpt
		if inpt.ndim == 3:
			h, w, n = inpt.shape
		elif inpt.ndim == 2:
			h, w = inpt.shape
			n = 1
		else:
			raise Exception("Expected ndim 2 or 3, received " + str(image.ndim))
		
		assert n == self.f_n, "Expected kernel of depth {}, got {}".format(n, self.f_n)
		
		if self.padding == "same":
			p = int(((h - 1) * self.strides + self.f_h - h) / 2)
		elif self.padding == "valid":
			p = 0

		out_h = ((h - self.f_h + 2 * p) / self.strides) + 1
		out_w = ((w - self.f_w + 2 * p) / self.strides) + 1

		assert out_h.is_integer(), "Kernel dimensions does not match"

		out_h = int(out_h)
		out_w = int(out_w)
		
		inpt = np.pad(inpt, ((p, p), (p, p), (0, 0)), mode="constant") if inpt.ndim == 3 else np.pad(inpt, p, mode="constant")
		output = np.zeros((out_h, out_w, self.num_filters))

		for i, j, im_region in self.iterate_regions(inpt):
			for f in range(self.num_filters):
				output[i, j, f] = np.sum(im_region * self.filters[f])
		return output
	
	def backprop(self, dl_dout, alpha):
		dl_dfilters = np.zeros(self.filters.shape)
		dl_dinputs = np.zeros(self.inpt.shape)

		for i, j, im_region in self.iterate_regions(self.inpt):
			for k in np.arange(self.num_filters):
				dl_dfilters[k] += dl_dout[i, j, k] * im_region

		for i, j, im_region in self.iterate_regions(dl_dinputs):
			for k in np.arange(self.num_filters):
				dl_dinputs[i : i + self.f_h, j : j + self.f_w] += dl_dout[i, j, k] * self.filters[k]
		
		self.filters += -alpha * dl_dfilters
		
		return dl_dinputs
	
class MaxPool2D:
	def __init__(self, filter_shape, strides=1):
		self.f_h = filter_shape[0]
		self.f_w = filter_shape[1]
		self.strides = strides
	
	def __repr__(self):
		return "MaxPool2D ({}, {})".format(self.f_h, self.f_w)

	def iterate_regions(self, image):
		h, w = image.shape[0], image.shape[1]
		for i in range(0, h - self.f_h + 1, self.strides):
			for j in range(0, w - self.f_w + 1, self.strides):
				im_region = image[i : i + self.f_h, j : j + self.f_w]
				yield i // self.strides, j // self.strides, im_region
	
	def forward(self, inpt):
		self.inpt = inpt
		self.inpt_shape = inpt.shape

		if inpt.ndim == 3:
			h, w, n = inpt.shape
		elif inpt.ndim == 2:
			h, w = inpt.shape
			n = 1
		else:
			raise Exception("Expected ndim 2 or 3, received " + str(inpt.ndim))
		
		out_h = ((h - self.f_h) / self.strides) + 1
		out_w = ((w - self.f_w) / self.strides) + 1

		assert out_h.is_integer(), "Kernel dimensions does not match"

		out_h = int(out_h)
		out_w = int(out_w)

		output = np.zeros((out_h, out_w, n))
		for i, j, im_region in self.iterate_regions(inpt):
			output[i, j] = np.amax(im_region, axis=(0, 1))
		
		return output
	
	def backprop(self, dl_dout, alpha):
		dl_dinputs = np.zeros(self.inpt_shape)
		
		for i, j, im_region in self.iterate_regions(self.inpt):
			h, w, f = im_region.shape
			amax = np.amax(im_region, axis=(0, 1))

			for i2 in np.arange(h):
				for j2 in np.arange(w):
					for k in np.arange(f):
						if im_region[i2, j2, k] == amax[k]:
							dl_dinputs[i + i2, j + j2, k] = dl_dout[i, j, k]
		return dl_dinputs

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

class LogLoss:
	@staticmethod
	def calculate_loss(x, y):
		return -np.log(x[y])
	
	@staticmethod
	def derivative(x, y):
		grad = np.zeros(len(x))
		grad[y] = -1 / x[y]
		return grad

class Sigmoid:
	@staticmethod
	def fn(x):
		return 1. / (1 + np.exp(-x))
	
	@staticmethod
	def derivative(x):
		return x * (1 - x)
	
class Relu:
	@staticmethod
	def fn(x):
		return np.maximum(0, x)
	
	@staticmethod
	def derivative(x):
		return (x > 0) * 1

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

trainX = mnist.train_images()[:1000] / 255.0
trainY = mnist.train_labels()[:1000]
testX = mnist.test_images()[:1000] / 255.0
testY = mnist.test_labels()[:1000]

permutation = np.random.permutation(len(trainX))
trainX = trainX[permutation]
trainY = trainY[permutation]

conv1 = Conv2D(32, (3, 3), padding="valid")
conv2 = Conv2D(64, (3, 3, 32), padding="valid")
pool1 = MaxPool2D((2, 2), strides=2)
dense1 = Dense(12 * 12 * 64, 128, Relu)
softmax1 = Softmax(128, 10)

model = Model(LogLoss, conv1, conv2, pool1, dense1, softmax1)
print(model)
model.fit(trainX, trainY, 5, 0.005)
model.validate(testX, testY)
