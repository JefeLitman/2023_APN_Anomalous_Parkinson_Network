"""This file contain the methodology and stept to run the training and inference for the base model of Ganomaly 2D. This file contain specifically the preprocessing methodology and the running loop.
Version: 0.2
Made by: Edgar Rangel
"""

import gc
import random
import numpy as np
import tensorflow as tf

from datasets.dict_features import get_ganomaly
from utils.metrics import *
from utils.savers import save_videos, save_latent_vectors

from models.ganomaly.model import get_2D_models
from models.ganomaly.utils.losses import *
from models.ganomaly.utils.processing import *
from models.ganomaly.utils.weights_init import reinit_model
from models.ganomaly.utils.exp_docs import *
from models.ganomaly.utils.printers import print_metrics
from models.ganomaly.utils.savers import save_errors, save_frames

if os.getenv("CUDA_VISIBLE_DEVICES") != '-1':
    gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.debugging.set_log_device_placement(False)

N_CPUS = opts.n_cpus
dataset_path = opts.dataset_path
encoding_dictionary = get_ganomaly()
encoding_dictionary