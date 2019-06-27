import keras
from keras.datasets import mnist
from keras.callbacks import Callback

import os
import numpy as np
            
def get_split(nb,x,y):
    '''
    divides data in two sets, one with nb samples per class, the other contains the rest
    '''
    train_selection = set()
    test_selection = set()
    for c in range(y.shape[1]): # iterates on classes
        indexes_class = np.where(y[:,c]==1)[0]
        
        train_selection = train_selection.union(set(np.random.choice(indexes_class,size = nb,replace = False)))
        test_selection  = test_selection.union(set(indexes_class).difference(train_selection))
        
    train_selection = list(train_selection)
    test_selection  = list(test_selection)
    
    x_train,x_test = x[train_selection], x[test_selection]
    y_train,y_test = y[train_selection], y[test_selection]
    
    return x_train, y_train, x_test, y_test

def load_reduced_mnist_data(samples_per_class = 1000):
    '''
    Experiment is run on a fraction of the mnist data.
    Reduced dataset is stored for reproducibility.
    
    samples_per_class is the number of samples per class that are used for training
    '''
    if not os.path.isfile('x_train_reduced.npy'):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        m, st = x_train.mean(), x_train.std()
        x_train -=m
        x_test -=m
        x_train /=st
        x_test /=st

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        x_train, y_train, _,_ = get_split(samples_per_class,x_train,y_train)

        np.save('x_train_reduced.npy',x_train)
        np.save('y_train_reduced.npy',y_train)
        np.save('x_test.npy',x_test)
        np.save('y_test.npy',y_test)

        x_train = x_train.reshape(x_train.shape[0], 784)
        x_test = x_test.reshape(x_test.shape[0], 784)
    else:
        x_train=np.load('x_train_reduced.npy')
        y_train=np.load('y_train_reduced.npy')
        x_test=np.load('x_test.npy')
        y_test=np.load('y_test.npy')

        x_train = x_train.reshape(x_train.shape[0], 784)
        x_test = x_test.reshape(x_test.shape[0], 784)
        
    return x_train, y_train, x_test, y_test

#class StoppingCriteria(Callback):
#    '''
#    Callback that stops training before the announced number of epochs when some criteria are met.
#    '''
#    def __init__(self, not_working=(0.,-1), finished = 1.1, converged = np.inf):
#        '''
#        not_working is a tuple (acc,nbepochs) with the accuracy that should be reached after nbepochs to consider the training as working
#        finished is a training accuracy value for which the training can be considered as finished
#        converged is the number of epochs with unchanged training loss which indicates that the network doesn't change anymore
#        '''
#        super().__init__()
#        self.acc, self.nbepochs = not_working
#        self.finished = finished
#        self.converged = converged
#        
#        self.previous_loss = -1
#        self.counter = 0
#        
#        
#    def on_epoch_end(self, epoch, logs=None):
#        if epoch ==self.nbepochs and logs.get('acc')<= self.acc:
#            self.model.stop_training = True
#        
#        if logs.get('acc')>=self.finished:
#            self.model.stop_training = True
#        
#        if logs.get('loss') == self.previous_loss:
#            self.counter += 1
#            if self.counter >= self.converged:
#                self.model.stop_training = True
#        else:
#            self.counter = 0
#            self.previous_loss = logs.get('loss')