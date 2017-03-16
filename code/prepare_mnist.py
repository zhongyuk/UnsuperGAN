import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from preprocess import *

def prepare_mnist2(data_filename, cls=[3,6]):
	data = pd.read_csv(data_filename)
	y = data[['label']]
	X = data[data.columns[1:]]
	X = X.as_matrix()
	# zero-mean and zca-whitening
	X_centered = center_data(X)
	#X_whitened = zca_whiten(X_centered)
	#X_whitened = preprocess_data(X)
	#X_preprocessed = X_whitened.astype(np.float32)
	X_preprocessed = X_centered.astype(np.float32)

	idx = [(y==c).values for c in cls]
	cls_idx = np.logical_or(idx[0], idx[1])
	cls_idx = np.squeeze(cls_idx)
	cls_X = X_preprocessed[cls_idx, :]
	cls_X = np.reshape(cls_X, [-1, 28, 28, 1])
	y = y.as_matrix().astype(np.float32)
	cls_y = y[cls_idx, :]
	cls_y = np.arange(2)==cls_y[:, None]
	cls_y = cls_y.astype(np.float32)
	cls_y = np.squeeze(cls_y, axis=1)
	return cls_X, cls_y