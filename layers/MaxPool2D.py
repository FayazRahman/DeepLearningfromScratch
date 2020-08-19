import numpy as np

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

