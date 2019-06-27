import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import History, Callback

def one_hot_encoding(labels,nb_labels = None):
    '''
    One-hot encoding of labels
    '''
    n = len(labels)
    
    if not nb_labels:
        nb_labels = labels.max() + 1
        if nb_labels != len(np.unique(labels)):
            warnings.warn('Bad numbering of labels. '+
                          'Or some labels are not present in dataset.\n'+
                          'Number of classes: '+str(nb_labels)+'.\n'
                          'Number of classes present: '+str(len(np.unique(labels)))+'.')
    
    y = np.zeros((n,nb_labels), dtype = bool)
    y[np.arange(n),labels.flatten()] = 1
    return y

def get_val_split(x,y,validation_split):
    '''
    Validation set creation
    validation_split is the portion of data used for validation (in [0,1])
    '''
    if validation_split == 0.:
        return [x,y], None
    elif validation_split >0. and validation_split <=1:
        index = np.random.permutation(x.shape[0])
        train_index = index[int(validation_split*x.shape[0]):]
        val_index =   index[:int(validation_split*x.shape[0])]
        return [x[train_index], y[train_index]], [x[val_index], y[val_index]]
    else:
        raise ValueError("The value given to validation_split is not between 0 and 1")

def lr_schedule(initial_lr,factor,epochs):
    '''
    utility for creating a learning rate schedule that drops the learning rate by some factor at the specified epochs
    epoch indices starts at 0
    '''
    def schedule(epoch):
        lr = initial_lr

        for i in range(len(epochs)):
            if epoch>= epochs[i]:
                lr *= factor
        return lr
    return schedule

colors = 'bgrcmykw'

def plot_history(history, metric_names = None, title = None, ignore_epochs = False, new_figure = True, val = True):
    """ 
    Plot the history returned by Keras' fit function
    
    metrics_names is a list containing the metrics/costs to be displayed
    If no metrics specified, model.metrics_names is used
    It uses one color per metric, dashed lines for validation set
    
    history is either a keras History object or a dictionary with keys 'epoch' and 'history'.
    """    
    if isinstance(history,dict):
        epoch = history['epoch']
        history = history['history']
    elif isinstance(history, History):
        epoch = history.epoch
        history = history.history
    else:
        raise ValueError()
        
    
    if not metric_names:
        metric_names = [metric for metric in history.keys() if metric[:4] != 'val_']
    
    if ignore_epochs:
        epoch = range(len(history[metric_names[0]]))
    
    if new_figure:
        plt.figure()
    
    handles = [None]*len(metric_names)
    for i,metric in enumerate(metric_names):
        handles[i], = plt.plot(epoch,history[metric], '-'+colors[i],label = metric)
        if val and 'val_'+metric in history.keys():
            plt.plot(epoch,history['val_'+metric],'--'+colors[i])
        
    plt.legend(handles = handles)
    plt.xlabel('epoch')
    
    if title:
        plt.title(title)

def compare_histories(histories, metric_name, names, val = False, title = None):
    """ 
    Plots evolution of metric_name on training and eventually validation set for
    the histories in histories list.
    
    It uses one color per history, dashed lines for validation set
    A legend is added with names associated with histories

    A history is either a keras History object or a dictionary with keys 'epoch' and 'history'.
    """    
    for i,history in enumerate(histories):
        if isinstance(history, History):
            histories[i] = history_todict(history)
    
    plt.figure()
    
    handles = [None]*len(histories)
    for i,history in enumerate(histories):
        handles[i], = plt.plot(history['epoch'],history['history'][metric_name], 
                               '-'+colors[i],label = names[i])
        if val:
            plt.plot(history['epoch'],history['history']['val_'+metric_name],'--'+colors[i])
    
    plt.legend(handles = handles)
    plt.xlabel('epoch')
    if metric_name[-3:] == 'acc':
        plt.ylabel('accuracy')
    else:
        plt.ylabel(metric_name)
    
    if title:
        plt.title(title)
        
        
def history_todict(history):
    """
    returns python dictionary containing keras history object content.
    Easier to handle for pickle
    """
    if hasattr(history,'epoch'):
        return {'epoch' : history.epoch,'history' : history.history}
    elif hasattr(history,'batch'):
        return {'batch' : history.batch,'history' : history.history}

class History_batch(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    
    recordings are made after every batch
    """

    def on_train_begin(self, logs=None):
        self.batch = []
        self.history = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch.append(batch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            