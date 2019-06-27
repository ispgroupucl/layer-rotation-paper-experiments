import matplotlib
import matplotlib.pyplot as plt

import numpy as np

def visualize_1stlayer_weights(weights,neuron_indices = None, nb_neurons = 5, neurons_per_line = 10, axes = None):
    '''
    Tool for visualizing weights associated to hidden neurons of the 1-layer MLP
    assumes 'channels last' ordering: spatial-x, spatial-y, input depth, output depth 
    '''
    if not neuron_indices:
        neuron_indices = range(nb_neurons) # if no indices specified, visualizes first neurons only
    
    if axes is None:
        f, axes = plt.subplots(int(len(neuron_indices)/neurons_per_line), neurons_per_line, 
                               figsize = (neurons_per_line*3,int(len(neuron_indices)/neurons_per_line)*3))
    axes = np.array([axes]) if len(axes.shape) == 1 else axes
    for i,j in enumerate(neuron_indices):
        line = i//neurons_per_line
        column = i%neurons_per_line
        
        # select neuron
        index = [slice(None)]*len(weights.shape)
        index[-1] = j
        
        # convert to image
        kernel = weights[index].reshape((28,28))
#         kernel = (kernel - np.min(weights[index])) / np.max(weights[index])
        
        axes[line,column].imshow(kernel, interpolation = 'none')
        
        axes[line,column].set_xticks([])
        axes[line,column].set_yticks([])