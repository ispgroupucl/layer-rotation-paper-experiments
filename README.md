# An experimental study of layer-level training speed and its impact on generalization
This repository contains the code used to create all the figures of our paper "An experimental study of layer-level training speed and its impact on generalization", by Simon Carbonnelle and Christophe De Vleeschouwer, submitted at ICLR ([OpenReview link](https://openreview.net/forum?id=HkeILsRqFQ)).  

### Code structure
Code structure tries to follow the structure of the paper. Name of folders correspond to the name of sections of the paper, to the exception of '/MNIST toy example' which refers to Figure 1 presented in the Introduction.

Files are in the parent folder if they are used in multiple sections:  
The notebooks train_on_baseline_tasks and train_on_sot_tasks contain the code for training on the (C10-CNN1,C100-resnet,tiny-CNN) tasks and (C10-CNN2,C100-WRN) tasks respectively (cfr. Table 1 of the paper), which are used across multiple sections.

import_task.py and models.py are used to load the data and the untrained models corresponding to the 5 tasks used in the experiments.

rotation_rate_utils.py contains the code for creating layer-wise angle deviation curves visualizations

layca_optimizers.py contains the code to apply Layca on SGD, Adam, RMSprop or Adagrad, and to use layer-wise learning rate multipliers when using SGD.

### Libraries configuration
Code was run with tensorflow-gpu 1.4.0 and keras 2.1.2
