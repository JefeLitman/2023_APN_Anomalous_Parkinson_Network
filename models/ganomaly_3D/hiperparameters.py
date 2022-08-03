"""This file contains all the hiperparameters for GANomaly 3D model. You can modify this file in order to change default values stablished here.
Version: 1.5
Made by: Edgar Rangel
"""

def get_options():
    """This function return a dictionary with the hiperparameters and options to execute the GANomaly 3D model in any of the selected modes. It doesn't require any parameter."""

    opts = dict(
        gpus = '0', # ID of the GPU which will be used
        n_cpus = 16, # Number of CPU cores to use while running
        lr = 0.0002, # Learning rate
        dataset_path = "./datasets/gait_v2/gait_v2.tfrecord", # Absolute path where the tfrecord is located to be used
        normal_class = 1, # Class label that will be the normal data in the training process
        kfolds = 5, # Number of kfolds in which the model will be evaluated with the tfrecord
        batch_size = 16, # Input batch size
        epochs = 20000, # Quantity of epochs to do in training
        seed = 8128, # Seed used to enable the replicability of the experiment
        save_path = "./results", # Path where the experiments will be saved
        save_frecuency = 1000, # Integer indicating between how many epochs the results and models will be saved
        gen_model_path = "./results", # Path where the generator model (h5) is allocated and will be loaded to run trained models
        disc_model_path = "./results", # Path where the discriminator model (h5) is allocated and will be loaded to run trained models
        eval_train = True, # If its True, then the loaded model will evaluate train data and test data together.
        isize = 64, # Input size of the videos, e.g. 64 equals to videos with shape 64x64x64
        nc = 1, # Quantity of channels in the data
        nz = 100, # Context vector size
        ngf = 64, # Quantity of initial filters in the first convolution of the encoder
        extra_layers = 0, # Quantity of layer blocks to add before reduction
        w_adv = 1, # Adversarial loss weight
        w_con = 50, # Contextual loss weight
        w_enc = 1, # Encoder loss weight
        beta_1 = 0.5, # Momentum of beta 1 in adam optimizer for generator and discriminator
        beta_2 = 0.999, # Momentum of beta 2 in adam optimizer for generator and discriminator
        readme = """This file contains information about the experiment made in this instance.

All models saved don't include the optimizer, but this file explains how to train in the same conditions.

Basic notation:

- {i}_Ganomaly3D-{size}x{size}x{size}x{nc}: Experiment id, name of the model and input dimension of model.
- H x W x F, F x H x W x C or H x W x C: Data dimensions used where F are frames, H height, W width and C channels.

Experiment settings:
- The seed used was {seed} for python random module, numpy random and tf random after the library importations.
- The batch size was of {batch}.
- The optimizer used in this experiment was Adam for generator and discriminator.
- The number of classes in this dataset are 2 (Normal and Parkinson) .
- This experiment use the data of gait_v2/dataset_09-jun-2022 tfrecord.
- The initial lr was of {lr}.
- The beta 1 and beta 2 for adam optimizer was {beta_1} and {beta_2} respectively.
- The total epochs made in this experiment was of {epochs}.
- The context vector size (nz) was of {nz}.
- The # channels in data (nc) was of {nc}.
- The initial filters in the first convolution of the encoder was {ngf}.
- The quantity of layer blocks to add before reduction was of {extra_layers}.
- The weights for adversarial, contextual and encoder error respectively in generator were {w_gen}.

Transformations applied to data (following this order):
- Resize: We resize the frames of volumes to H x W ({size} x {size}).
- Equidistant Oversampling volume: We take {size} frames sampled equidistant of volumes to train and test the data.
- Convert: We convert the videos in RGB to Grayscale.
- Normalize: We normalize the volume with mean and std of 0.5 for both.
- Scale: We scale the data between -1 and 1 using min max scaler to be comparable with generated images.
- Identify: We identify each video per patient with an integer value.
- Randomize: We randomize the order of samples in every epoch.

Training process:
- The data doesn't have train and test partition but we make the partitions like this:
    * ~80% (11 patients) of normal (parkinson) data is used in train for kfold {k}.
    * ~20% (3 patients) of normal (parkinson) data is used in test for kfold {k}.
    * ~20% (3 patients) of abnormal (healthy) data are used in test for kfold {k}.
"""
    )

    opts["w_gen"] = (opts["w_adv"], opts["w_con"], opts["w_enc"])
    
    return opts