"""This is the main file to run the differents models and experiments in this repository for Anomalous Parkinson Gait research.
Version: 1.1
Made by: Edgar Rangel
"""

import os
import argparse

def run():
    """Wrapper function to run the experiments by command line. It doesn't require anything but only to pass two parameters on it. You can also run the experiments using the notebooks if you feel more confortable with them.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='ganomaly_3D', help='Name of the model to use. It must be equal to filename (without .py) in scripts folder. e.g. ganomaly_3D')
    parser.add_argument('--mode', type=str, default='train_eval', help='In which mode the model selected will be running. It must be equal to a filename (without .py) of the modes folder in the model. e.g. "train_eval"')

    parsed_args = parser.parse_args()

    from scripts.ganomaly_3D import run as ganomaly_3D
    from scripts.ganomaly import run as ganomaly

    locals()[parsed_args.model](parsed_args.mode)

if __name__ == '__main__':
    run()