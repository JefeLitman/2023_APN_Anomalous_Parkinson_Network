"""This file contains all the process to do the data preprocessing for GANomaly model. You must modify this file in order to change the preprocessing developed here.
https://arxiv.org/abs/1805.06725
Version: 1.0
Made by: Edgar Rangel
"""

import tensorflow as tf
from .utils.processing import resize, oversampling_equidistant_full, rgb_to_grayscale, normalize_accros_channels, min_max_scaler, repeat_and_identify_frames, add_video_id

def preprocess_gait_dataset(raw_data, opts, normal_patients_ids, abnormal_patients_ids):
    """Function that develop all the preprocessing necessary over the tfrecord for gait_v2. It returns two elements in the following order: a list with the normal patients instances of tf.data.Dataset and another for abnormal data.
    Args:
        raw_data (tf.data.Dataset): An instance of tf.data.Dataset object which contain all the data and must return for each sample a video, label and patient_id in that order.
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        normal_patients_ids (List[Int]): A list or array of integers indicating which patients are from control population.
        abnormal_patients_ids (List[Int]): A list or array of integers indicating which patients are from parkinson population.
    """
    if opts["nc"] == 1:
        rgb_2_bw = True
    elif opts["nc"] == 3:
        rgb_2_bw = False
    else:
        raise AssertionError('The only possible channel number allowed are RGB (3) or Black & White (1) in the preprocessing of GANomaly Model')

    normal_class = opts["normal_class"]
    if normal_class == 0:
        abnormal_class = 1
    elif normal_class == 1:
        abnormal_class = 0
    else:
        raise AssertionError('The only classes allowed in BivL2ab Gait Dataset are control (0) and parkinson (1). Class passed {}'.format(normal_class))

    normal_data = raw_data.filter(lambda x,y,z: tf.equal(y, normal_class))
    abnormal_data = raw_data.filter(lambda x,y,z: tf.equal(y, abnormal_class))
    frames = opts["isize"]
    N_CPUS = opts["n_cpus"]

    normal_patients = []
    for i in normal_patients_ids:
        data = normal_data.filter(
            lambda x, y, z: tf.equal(z, i)
        ).map(
            lambda x, y, z: resize(x, y, [frames, frames], z), N_CPUS
        ).flat_map(
            lambda x, y, z: oversampling_equidistant_full(x, y, frames, z)
        )
        if rgb_2_bw:
            data = data.map(rgb_to_grayscale, N_CPUS)
        
        data = data.map(
            lambda x, y, z: normalize_accros_channels(x, y, 0.5, 0.5, z), N_CPUS
        ).map(
            lambda x, y, z: min_max_scaler(x, y, -1., 1., z), N_CPUS
        ).map(
            repeat_and_identify_frames, N_CPUS
        ).enumerate().map(
            add_video_id, N_CPUS
        ).unbatch().cache()

        normal_patients.append(data)
        
    abnormal_patients = []
    for i in abnormal_patients_ids:
        data = abnormal_data.filter(
            lambda x, y, z: tf.equal(z, i)
        ).map(
            lambda x, y, z: resize(x, y, [frames, frames], z), N_CPUS
        ).flat_map(
            lambda x, y, z: oversampling_equidistant_full(x, y, frames, z)
        )
        if rgb_2_bw:
            data = data.map(rgb_to_grayscale, N_CPUS)
        
        data = data.map(
            lambda x, y, z: normalize_accros_channels(x, y, 0.5, 0.5, z), N_CPUS
        ).map(
            lambda x, y, z: min_max_scaler(x, y, -1., 1., z), N_CPUS
        ).map(
            repeat_and_identify_frames, N_CPUS
        ).enumerate().map(
            add_video_id, N_CPUS
        ).unbatch().cache()

        abnormal_patients.append(data)

    return normal_patients, abnormal_patients