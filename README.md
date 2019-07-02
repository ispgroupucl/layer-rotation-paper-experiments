# Layer rotation: a surprisingly powerful indicator of generalization in deep networks?
This repository contains the code used to create all the figures of our paper "Layer rotation: a surprisingly powerful indicator of generalization in deep networks?", by Simon Carbonnelle and Christophe De Vleeschouwer, available [on arXiv](https://arxiv.org/abs/1806.01603v2) and presented during the ICML 2019 workshop "Identifying and Understanding Deep Learning Phenomena".

### Code structure
Code structure tries to follow the structure of the paper. I.e.:
- 'A systematic study of layer rotation configurations with Layca' = Section 4
- 'A study of layer rotation in standard training settings' = Section 5
- 'MNIST - 1st layer feature visualization' = Section 6


import_task.py and models.py are used to load the data and the untrained models corresponding to the 5 tasks used in the experiments.

rotation_rate_utils.py contains the code for recording and visualizing layer rotation curves

layca_optimizers.py contains the code to apply Layca on SGD, Adam, RMSprop or Adagrad, and to use layer-wise learning rate multipliers when using SGD.

get_training_utils.py contains utilities to get (optimal) training parameters such as learning rate schedules, learning rate multipliers, stopping criteria and optimizers for training of the five tasks

experiment_utils.py contains utilities such as training curves visualization, one-hot encoding, ...

### Libraries configuration
Code was run with tensorflow-gpu 1.4.0 and keras 2.1.2
