import numpy as np


from keras.callbacks import Callback
import keras.backend as K

class LearningRateScheduler_with_warmup(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        warmup_type: either 'discrete' (sudden transition towards high learning rate) or 'continuous' (linear transition)
        warmup_period: duration (in epochs) of the warmup
        steps_per_epoch: number of batches per epoch (only used if warmup_type = 'continuous')
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, warmup_type = 'discrete', warmup_period = 1,steps_per_epoch = 0,verbose=0):
        super().__init__()
        self.schedule = schedule
        self.warmup_type = warmup_type
        self.warmup_period = warmup_period
        self.verbose = verbose
        
        if warmup_type == 'continuous':
            self.steps_per_epoch = steps_per_epoch
        
    def on_train_begin(self, logs={}):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
    def on_batch_begin(self, batch, logs = None):
        if self.warmup_type == 'continuous' and self.epoch<self.warmup_period:            
            lr = self.schedule(self.epoch)
           
            factor = (self.epoch * self.steps_per_epoch + batch)/ (self.warmup_period * self.steps_per_epoch)
            
            K.set_value(self.model.optimizer.lr, lr*factor+lr/10.*(1-factor))            

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch=epoch
        
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
            
        if self.warmup_type =='discrete' and epoch<self.warmup_period:
            K.set_value(self.model.optimizer.lr, lr/10.)
        else:
            K.set_value(self.model.optimizer.lr, lr)
            
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)