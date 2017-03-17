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

def visulize(rcdfn, save_imgfn, imgsid):
	data = unpickle(rcdfn)
	images = data['img_rcd'][imgsid]
	xreals = images['Xreal']
	xfakes = images['generated']
	plot_GDimgs(xreals, xfakes, save_imgfn)

def test_plot_GDimgs():
	data = unpickle("rcd2")
	images = data['img_rcd'][-1]
	xreals = images['Xreal']
	xfakes = images['generated']
	plot_GDimgs(xreals, xfakes, "testimg")

def plot_loss(record):
    G_loss = record['Grcd']
    D_loss = record['Drcd']
    epoches = np.array(range(G_loss.shape[0]))
    plt.plot(epoches, G_loss, label='Generator Loss')
    plt.plot(epoches, D_loss, label='Discriminator Loss')
    plt.xlim([0, G_loss.shape[0]+3])
    plt.legend(loc='upper right')
    plt.show()

def plot_pred_acc(record):
    src_pred = rcd['src_pred']
    cls_pred = rcd['cls_pred']
    src_pred_fake_in = src_pred['fake_in']
    src_pred_real_in = src_pred['real_in']
    cls_pred_fake_in = cls_pred['fake_in']
    cls_pred_real_in = cls_pred['real_in']
    epoches = np.array(range(src_pred_fake_in.shape[0]))
    plt.subplot(211)
    plt.plot(epoches, src_pred_fake_in, label='Source Prediction - fake input')
    plt.plot(epoches, src_pred_real_in, label='Source Prediction - real input')
    plt.legend(loc='lower right'); 
    plt.ylim([0., 1.1]);plt.show();
    plt.subplot(212)
    plt.plot(epoches, cls_pred_fake_in, label='Class Prediction - fake input')
    plt.plot(epoches, cls_pred_real_in, label='Class Prediction - real input')
    plt.legend(loc='lower right')
    plt.ylim([0., 1.1]); plt.show()

if __name__=='__main__':
	#test_plot_GDimgs()
	rcdfn = 'train_rcd/rcd3'
	savename = 'rcd3_samples1'
	visulize(rcdfn, savename, imgsid=25)
