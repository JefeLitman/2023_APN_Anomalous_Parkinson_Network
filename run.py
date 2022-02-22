"""This is the main file to run the differents models and experiments for
Anomalous Parkinson Gait research.
Version: 0.0.1
Made by: Edgar Rangel
"""

import os
import utils.options as uo
import scripts.ganomaly as ganomaly

def run():
    """Wrapper function to run the experiments by command line.
    You can also run the experiments using the notebooks if you feel
    more confortable with them
    """

    opts = uo.get_parsed_parameters()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpus

if __name__ == '__main__':
    run()