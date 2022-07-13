"""This file contains all the hiperparameters for GANomaly 3D model. You can modify this file in order to change default values stablished here.
Version: 1.2
Made by: Edgar Rangel
"""

def get_options():
    """This function return a dictionary with the hiperparameters and options to execute the GANomaly 3D model in any of the selected modes. It doesn't require any parameter."""

    opts = dict(
        gpus = '0', # ID of the GPU which will be used
        n_cpus = 16, # Number of CPU cores to use while running
        lr = 0.0002, # Learning rate
        dataset_path = "./datasets/gait_v2/gait_v2.tfrecord", # Absolute path where the tfrecord is located to be used
        normal_class = 0, # Class label that will be the normal data in the training process
        kfolds = 5, # Number of kfolds in which the model will be evaluated with the tfrecord
        batch_size = 16, # Input batch size
        epochs = 20000, # Quantity of epochs to do in training
        seed = 8128, # Seed used to enable the replicability of the experiment
        save_path = "./results", # Path where the experiments will be saved
        isize = 64, # Input size of the videos, e.g. 64 equals to videos with shape 64x64x64
        nc = 1, # Quantity of channels in the data
        nz = 100, # Context vector size
        ngf = 64, # Quantity of initial filters in the first convolution of the encoder
        extra_layers = 0, # Quantity of layer blocks to add before reduction
        w_adv = 1, # Adversarial loss weight
        w_con = 50, # Contextual loss weight
        w_enc = 1, # Encoder loss weight
        beta_1 = 0.5, # Momentum of beta 1 in adam optimizer for generator and discriminator
        beta_2 = 0.999 # Momentum of beta 2 in adam optimizer for generator and discriminator
    )

    opts["w_gen"] = (opts["w_adv"], opts["w_con"], opts["w_enc"])
    
    return opts