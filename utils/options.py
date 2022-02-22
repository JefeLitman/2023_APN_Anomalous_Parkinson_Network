"""This file contain the options for differents experiments and parameters
required by each model of Anomalous Parkinson Gait research.
Version: 1.0
Made by: Edgar Rangel
"""

import argparse


def get_parsed_parameters():
    """This function creates the parser and parse the parameters given in the main file
    to run the experiments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ################################# GENERAL PARAMETERS #####################################################
    parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--n_cpus', type=int, help='number of cpu cores to use', default=16)
    parser.add_argument('--lr', type=float, help='learning rate for the model in learning', default=1e-3)
    parser.add_argument('--dataset_path', type=str, default='', help='Absolute path where the tfrecord is')
    parser.add_argument('--model', type=str, default='ganomaly', help='Name of the model to use. It must be equal to folder name in models. e.g. ganomaly')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--epochs', type=int, default=500, help='quantity of epochs to do in training')
    parser.add_argument('--val_4_epoch', type=int, default=1, help='calculate metrics after training each epoch: 1 | 0')
    parser.add_argument('--seed', type=int, default=8128, help='seed used to enable the replicability of the experiment')
    parser.add_argument('--save_path', type=str, default='', help='Absolute path where the experiment will be saved')
    parser.add_argument('--training', type=int, default=1, help='the experiment is for training a new model: 1 | 0')

    ################################# GANOMALY MODEL #########################################################
    parser.add_argument('--isize', type=int, default=64, help='nput size of the data (image or volume).')
    parser.add_argument('--nc', type=int, default=3, help='quantity of channels in the data')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent Zg vector')
    parser.add_argument('--ngf', type=int, default=64, help='quantity of layer blocks to add before reduction')
    parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--w_adv', type=float, default=1, help='adversarial loss weight')
    parser.add_argument('--w_con', type=float, default=50, help='contextual loss weight')
    parser.add_argument('--w_enc', type=float, default=1, help='encoder loss weight.')
    parser.add_argument('--model_dimension', type=str, default="2D", help='dimension of model to use in experiment: 2D | 3D')
    parser.add_argument('--beta_1', type=float, default=0.5, help='beta_1 term of adam')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 term of adam')

    ##########################################################################################################
    
    parsed_args = parser.parse_args()
    parsed_args.w_gen = (
        parsed_args.w_adv,
        parsed_args.w_con,
        parsed_args.w_enc,
    )

    return parsed_args
