"""This file contains methods to save elements for GANomaly 3D model.
Version: 1.5
Made by: Edgar Rangel
"""

import os
import numpy as np
import tensorflow as tf
from .processing import min_max_scaler
from .losses import l1_loss_batch, l2_loss_batch
from utils.savers import save_video, save_latent_vector, save_errors
from utils.common import format_index, get_partitions_paths, get_next_last_item

def save_videos(batch_videos, batch_labels, batch_ids, batch_videos_ids, folder_path, partition):
    """Function to save the batch of videos in the given folder path for train or test data, subdividing the normal and abnormal samples on different folders.
    Args:
        batch_videos (Array): A 5D array with (b, f, h, w, c) shape where b - batch size, f - frames, h - height, w - width and c - channels of the videos to be saved.
        batch_labels (Array): A 1D array with (b) shape with the labels of videos.
        batch_ids (Array): A 1D array with (b) shape with the patients ids of every video.
        batch_videos_ids (Array): A 1D array with (b) shape with the ids of videos in the batch.
        folder_path (String): The root path where the videos will be saved.
        partition (String): The partition to save the results, the available options are "train", "val" or "test".
    """
    assert batch_labels.shape[0] == batch_videos.shape[0] == batch_ids.shape[0] == batch_videos_ids.shape[0]
    if batch_videos.ndim != 5:
        raise AssertionError('The quantity of dimension for batch_videos must be 5 in for GANomaly 3D model')
            
    normal_path, abnormal_path = get_partitions_paths(folder_path, partition)
    if not os.path.isdir(normal_path):
        os.mkdir(normal_path)
    if not os.path.isdir(abnormal_path):
        os.mkdir(abnormal_path)

    for i, video in enumerate(batch_videos):
        if batch_labels[i] == 0:
            save_path =  normal_path
        elif batch_labels[i] == 1:
            save_path = abnormal_path
        else:
            raise AssertionError('There is an unknow label in the data. Label found {}, expected to be 0 or 1'.format(batch_labels[i]))

        folder = "{}_patient-{}_video-{}".format(
            format_index(get_next_last_item(save_path)),
            format_index(batch_ids[i]),
            format_index(batch_videos_ids[i])
        )
        save_path = os.path.join(save_path, folder)
        os.mkdir(save_path)
        save_video(video, save_path)

def save_latent_vectors(batch_latents, batch_labels, batch_ids, batch_videos_ids, folder_path, partition):
    """Function to save the batch of embeddings in the given folder path for train or test data, subdividing the normal and abnormal samples on different folders.
    Args:
        batch_latents (Array): A 2D numpy array with shape (b, z) where b = batch size and z = context vector size for the latent_vectors to be saved.
        batch_labels (Array): A 1D array with shape (b) for the labels of videos.
        batch_ids (Array): A 1D array with shape (b) for the patients ids of every video.
        batch_videos_ids (Array): A 1D array with shape (b) for the ids of videos in the batch.
        folder_path (String): The root path where the videos will be saved.
        partition (String): The partition to save the results, the available options are "train", "val" or "test".
    """
    assert batch_labels.shape[0] == batch_latents.shape[0] == batch_ids.shape[0] == batch_videos_ids.shape[0]
    if batch_latents.ndim != 2:
        raise AssertionError('The quantity of dimension for batch_latents must be 2 in for GANomaly 3D model')
            
    normal_path, abnormal_path = get_partitions_paths(folder_path, partition)
    if not os.path.isdir(normal_path):
        os.mkdir(normal_path)
    if not os.path.isdir(abnormal_path):
        os.mkdir(abnormal_path)

    for i, vector in enumerate(batch_latents):
        if batch_labels[i] == 0:
            save_path =  normal_path
        elif batch_labels[i] == 1:
            save_path = abnormal_path
        else:
            raise AssertionError('There is an unknow label in the data. Label found {}, expected to be 0 or 1'.format(batch_labels[i]))

        filename = "{}_patient-{}_video-{}".format(
            format_index(get_next_last_item(save_path)),
            format_index(batch_ids[i]),
            format_index(batch_videos_ids[i])
        )
        save_latent_vector(vector, save_path, filename)


def save_models(gen_model, disc_model, experiment_path, epoch = ""):
    """This function take the GANomaly 3D models and save them in the path given. Also it remove the previous models in the path to avoid saving a lot of innecessary files.
    Args:
        gen_model (Keras Model Instance): An instance of tf.keras.Model of the generator model to be saved.
        disc_model (Keras Model Instance): An instance of tf.keras.Model of the discriminator model to be saved.
        experiment_path (String): A string with the folder path in where the models will be saved.
        epoch (Integer): An optional integer indicating in which epoch the model reach while is training.
    """
    for i in sorted(os.listdir(experiment_path)):
        if "gen_model" in i:
            os.remove(os.path.join(experiment_path, i))
        elif "disc_model" in i:
            os.remove(os.path.join(experiment_path, i))
    if epoch != "":
        to_add = "_{}".format(epoch + 1)
    else:
        to_add = ""
    gen_model.save(os.path.join(experiment_path,"gen_model{}.h5".format(to_add)), 
        include_optimizer=False, save_format='h5')
    disc_model.save(os.path.join(experiment_path,"disc_model{}.h5".format(to_add)), 
        include_optimizer=False, save_format='h5')

def save_model_results(xyi, fake_images, latent_i, latent_o, feat_real, feat_fake, outputs_path, partition, clean_old):
    """This function take the given args (xyi, fake_images, latent_i, latent_o, feat_real, feat_fake) and save them in the outputs paths given for their own reason. This method doesn't return anything but save all the outputs for the model.
    Args:
        xyi (Tuple[Tensor]): A tuple with (videos, labels, patient_ids, videos_ids) elements in that respective order.
        fake_images (Tensor): A tensor with the generated videos by the model.
        latent_i (Tensor): A tensor with the latent vector Zg of the model.
        latent_o (Tensor): A tensor with the latent vector Z'g of the model.
        feat_real (Tensor): A tensor with the latent vector Zd of the model.
        feat_fake (Tensor): A tensor with the latent vector Z'd of the model.
        outputs_path (List[Str]): A list with the paths in which the model stores elements such as latent vectors, errors and videos.
        partition (String): The partition to save the results, the available options are "train", "val" or "test".
        clean_old (Boolean): Select True if you want to delete all the content in outputs folder or False otherwise.
    """
    if clean_old:
        for path in outputs_path:
            paths = get_partitions_paths(path, partition)
            for p in paths:
                if os.path.isfile(p + ".npy"):
                    os.system("rm {}".format(p + ".npy"))
                else:
                    os.system("rm -rf {}".format(os.path.join(p, "*")))

    save_latent_vectors(tf.squeeze(latent_i).numpy(), xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[0], partition)
    save_latent_vectors(tf.squeeze(latent_o).numpy(), xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[1], partition)
    save_latent_vectors(tf.reshape(feat_real, [xyi[0].shape[0], -1]).numpy(), xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[2], partition)
    save_latent_vectors(tf.reshape(feat_fake, [xyi[0].shape[0], -1]).numpy(), xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[3], partition)

    batch_videos = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in xyi[0]]].astype(np.uint8)
    save_videos(batch_videos, xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[4], partition)
    batch_videos = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in fake_images]].astype(np.uint8)
    save_videos(batch_videos, xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[5], partition)
    batch_videos = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in tf.abs(xyi[0] - fake_images)]].astype(np.uint8)
    save_videos(batch_videos, xyi[1].numpy(), xyi[2].numpy(), xyi[3].numpy(), outputs_path[6], partition)

    save_errors(l2_loss_batch(feat_real, feat_fake).numpy(), xyi[1].numpy(), outputs_path[7], partition)
    save_errors(l1_loss_batch(xyi[0], fake_images).numpy(), xyi[1].numpy(), outputs_path[8], partition)
    save_errors(l2_loss_batch(latent_i, latent_o).numpy(), xyi[1].numpy(), outputs_path[9], partition)