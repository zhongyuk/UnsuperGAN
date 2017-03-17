import numpy as np
from six.moves import cPickle as pickle
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
#sys.path.append("/home/paperspace/Documents/")
from preprocess import *

def unpickle(file):
    # Load pickled data
    fo = open(file, 'rb')
    dict = pickle.load(fo) 
    fo.close()
    return dict

def load_data(data_dir):
    # Load training and testing data
    train_fnroot = data_dir+'data_batch_'
    test_filename = data_dir+'test_batch'
    train_dataset = None
    test_dataset = None
    print("Loading the training data...")
    for i in range(1,6):
        train_filename = train_fnroot + str(i)
        batch = unpickle(train_filename)
        if i==1:
            train_dataset = batch['data']
            train_labels = np.array(batch['labels'])
        else:
            train_dataset = np.concatenate((train_dataset, batch['data']), axis=0)
            train_labels = np.concatenate((train_labels, batch['labels']))
    print("Loading the testing data...")
    test_batch = unpickle(test_filename)
    test_dataset = test_batch['data']
    test_labels = np.array(test_batch['labels'])
    return train_dataset, train_labels,  test_dataset, test_labels

def preprocess_data_cifar10(X, y, num_labels):
    # 1) One-hot encode labels
    # 2) Random Permute samples
    # 3) Change datatype to np.float32 to speed up
    # zero-mean and zca-whitening
    y_encoded = np.arange(num_labels)==y[:, None]
    perm = np.random.permutation(y_encoded.shape[0])
    X_preprocessed = X[perm]
    y_encoded = y_encoded[perm]
    return X_preprocessed.astype(np.float32), y_encoded.astype(np.float32)

def one_hot_encode(y, num_labels):
    '''One-Hot Encode labels'''
    y_encoded = np.arange(num_labels)==y[:, None]
    return y_encoded

def scale_input(x):
    '''scale input x to 0-1 range'''
    scaled_x = (255*np.ones_like(x)-x)/255.
    return scaled_x

def prepare_cifar10_input(data_dir, augmentation=False):
    # Load Data
    print("Load data", "."*32)
    train_dataset, train_labels, test_dataset, test_labels = load_data(data_dir)
    train_dataset, test_dataset = scale_input(train_dataset), scale_input(test_dataset)
    #train_dataset = whiten_data(train_dataset)
    #test_dataset = whiten_data(test_dataset)

    # Split 20% of training set as validation set
    #print("Split training and validation set", "."*32)
    #train_dataset, valid_dataset, train_labels, valid_labels = \
    #train_test_split(train_dataset, train_labels, test_size=5000,\
    #random_state=897, stratify=train_labels)
    # Print out data shapes
    print('Dataset\t\tFeatureShape\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    #print('Validation set:\t', valid_dataset.shape,'\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)

    # Reshape the data into pixel by pixel by RGB channels
    print("Reformat data", "."*32)
    train_dataset = np.rollaxis(train_dataset.reshape((-1,3,32,32)), 1, 4)
    #valid_dataset = np.rollaxis(valid_dataset.reshape((-1,3,32,32)), 1, 4)
    test_dataset = np.rollaxis(test_dataset.reshape((-1,3,32,32)), 1, 4)
    print('Dataset\t\tFeatureShape\t\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    #print('Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)

    # Dataset Parameters
    image_size = 32
    num_labels = 10
    num_channels = 3

    # Data Augmentation
    if augmentation:
        train_dataset, train_labels = augment_data_cifar10(train_dataset, train_labels)

    # Data Preprocess: change datatype; center the data
    print("Preprocess data", "."*32)
    train_dataset, train_labels = preprocess_data_cifar10(train_dataset, train_labels, num_labels)
    #valid_dataset, valid_labels = preprocess_data_cifar10(valid_dataset, valid_labels, num_labels)
    test_dataset,  test_labels  = preprocess_data_cifar10(test_dataset,  test_labels,  num_labels)
    #dataset_list = [train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels]
    dataset_list = [train_dataset, train_labels, test_dataset, test_dataset]
    print('Dataset\t\tFeatureShape\t\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    #print('Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)
    return dataset_list

if __name__=='__main__':
    data_dir = "/Users/Zhongyu/Documents/projects/CNNplayground/cifar10/data/"
    #data_dir = "/home/paperspace/Documents/cifar_data/"
    prepare_cifar10_input(data_dir)
