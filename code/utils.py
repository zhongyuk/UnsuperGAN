import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

def unpickle(file):
    # Load pickled data
    fo = open(file, 'rb')
    dict = pickle.load(fo) 
    fo.close()
    return dict

def rescale(img):
	max_array = 1.*np.ones_like(img)
	img_scaled = 255.* img/max_array
	return img_scaled.astype('uint8')

def plot_GDimgs(xreal, xfake, savename):
	sampler0 = rescale(xreal[0,:,:,:])
	samplef0 = rescale(xfake[0,:,:,:])
	mid = int(xreal.shape[0]/2)
	sampler1 = rescale(xreal[mid,:,:,:])
	samplef1 = rescale(xfake[mid,:,:,:])
	sampler2 = rescale(xreal[-1,:,:,:])
	samplef2 = rescale(xfake[-1,:,:,:])
	plt.subplot(321)
	plt.imshow(sampler0)
	plt.subplot(322)
	plt.imshow(samplef0)
	plt.subplot(323)
	plt.imshow(sampler1)
	plt.subplot(324)
	plt.imshow(samplef1)
	plt.subplot(325)
	plt.imshow(sampler2)
	plt.subplot(326)
	plt.imshow(samplef2)
	plt.savefig(savename+".png")

def visulize(rcdfn, save_imgfn):
	data = unpickle(rcdfn)
	images = data['img_rcd'][-1]
	xreals = images['Xreal']
	xfakes = images['generated']
	plot_GDimgs(xreals, xfakes, save_imgfn)

def test_plot_GDimgs():
	data = unpickle("rcd2")
	images = data['img_rcd'][-1]
	xreals = images['Xreal']
	xfakes = images['generated']
	plot_GDimgs(xreals, xfakes, "testimg")

if __name__=='__main__':
	test_plot_GDimgs()
