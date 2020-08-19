import numpy as np

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
