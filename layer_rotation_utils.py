'''
Utilities related to layer rotation
'''

import numpy as np
from scipy.spatial.distance import cosine

import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import Callback
import keras.backend as K
from keras.losses import categorical_crossentropy

def get_kernel_layer_names(model):
    '''
    collects name of all layers of a model that contain a kernel in topological order (input layers first).
    '''
    layer_names = []
    for l in model.layers:
        if len(l.weights) >0:
            if 'kernel' in l.weights[0].name:
                layer_names.append(l.name)
    return layer_names

def plot_layer_rotation_curves(deviations, ax = None):
    '''
    utility to plot the layer-wise cosine distances between current parameters and initial parameters, 
        as measured over training (i.e. layer rotation curves).
    deviations is a list of lists with epoch index in first axis, layer index in second axis, 
        containing the cosine distances for each layer as recorded over training
    '''
    distances = np.array(deviations)
    
    # get one color per layer
    cm = plt.get_cmap('viridis')
    cm_inputs = np.linspace(0,1,distances.shape[1])
    
    if not ax:
        ax = plt.subplot(1,1,1)
    for i in range(distances.shape[-1]):
        layer = i
        ax.plot(np.arange(distances.shape[0]+1), [0]+list(distances[:,layer]), label = str(layer), color = cm(cm_inputs[i]))

    ax.set_ylim([0,1.])
    ax.set_xlim([0,distances.shape[0]])
     
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine distance')

def compute_layer_rotation(current_model, initial_w):
    '''
    for each layer, computes cosine distance between current weights and initial weights
    initial_w is a list of tuples containing layer name and corresponding initial numpy weights
    '''
    s = []
    for l_name, w in initial_w:
        s.append(cosine( current_model.get_layer(l_name).get_weights()[0].flatten(), w.flatten()))
    return s

class LayerRotationCurves(Callback):
    '''
    Computes and saves layer rotation curves during training
    '''
    def __init__(self, batch_frequency=np.inf):
        '''
        batch_frequency is the frequency at which the cosine distances are computed (minimum once per epoch)
        '''
        super().__init__()
        self.batch_frequency = batch_frequency
        
        self.memory = []
    
    def set_model(self,model):
        super().set_model(model)
        layer_names = get_kernel_layer_names(model) 
        
        # initial_w is a list of tuples containing layer name and corresponding initial numpy weights
        self.initial_w = list(zip(layer_names,[model.get_layer(l).get_weights()[0] for l in layer_names]))
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0: #batch 0 is accepted, batch resets at 0 at every epoch

            dist = compute_layer_rotation(self.model, self.initial_w)

            self.memory.append(dist)
    
    def plot(self,ax = None):
        plot_layer_rotation_curves(self.memory,ax)
        
class StepwiseRotation(Callback):
    '''
    Computes and saves the tangens of the angle applied on each layer's weights during each training step
    '''
    def __init__(self):
        super().__init__()
        self.memory = []
    
    def set_model(self,model):
        super().set_model(model)
        self.layer_names = get_kernel_layer_names(model) 
    
    def on_batch_begin(self,batch,logs = None):
        # previous_w is a list of tuples containing layer name and corresponding initial numpy weights
        self.previous_w = list(zip(self.layer_names,[self.model.get_layer(l).get_weights()[0] for l in self.layer_names]))
        
    def on_batch_end(self, batch, logs=None):
        cos = compute_layer_rotation(self.model, self.previous_w)
        
        # converts cosinus distances in tangens of the angle
        norm = [np.sqrt(np.clip(1/(1-cosi)**2-1,a_min = 0.,a_max = None)) for cosi in cos] 
        # sometimes, due to numerical errors, cosi is negative, hence the clipping
        
        self.memory.append(norm)
        
class StepwiseLearningRateScheduler(Callback):
    """
    Callback for changing the layer-wise learning rates at each training step
    Is used to reproduce the layer rotations of a training procedure with Layca
    """

    def __init__(self, schedule, verbose=0):
        '''
        Schedule is a list of lists. First axis is batch index, second axis is layer index. 
        Provides learning rate for each batch/layer pair.
        '''
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
    
    def set_model(self,model):
        super().set_model(model)
        self.layer_names = get_kernel_layer_names(model)

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'multipliers'):
            raise ValueError('Optimizer must have a "multipliers" attribute.')
        
        self.batch_counter = 0
        
    def on_batch_begin(self, batch, logs=None):
        multipliers = self.model.optimizer.multipliers
        base_lr = float(K.get_value(self.model.optimizer.lr))
        
        for key,value in multipliers.items():
            layer_index = self.layer_names.index(key) # keys of multipliers dictionary should be the layer names
            K.set_value(multipliers[key], self.schedule[self.batch_counter][layer_index]/base_lr)
            
        self.batch_counter += 1
        