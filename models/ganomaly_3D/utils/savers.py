"""This file contains methods to save elements for GANomaly 3D model.
Version: 1.2
Made by: Edgar Rangel
"""

import os
from ....utils.savers import save_video, save_latent_vector
from ....utils.common import format_index, get_train_test_paths, get_next_last_item

def save_videos(batch_videos, batch_labels, batch_ids, batch_videos_ids, folder_path, training):
    """Function to save the batch of videos in the given folder path for train or test data, subdividing the normal and abnormal samples on different folders.
    Args:
        batch_videos (Array): A 5D array with (b, f, h, w, c) shape where b - batch size, f - frames, h - height, w - width and c - channels of the videos to be saved.
        batch_labels (Array): A 1D array with (b) shape with the labels of videos.
        batch_ids (Array): A 1D array with (b) shape with the patients ids of every video.
        batch_videos_ids (Array): A 1D array with (b) shape with the ids of videos in the batch.
        folder_path (String): The root path where the videos will be saved.
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_videos.shape[0] == batch_ids.shape[0] == batch_videos_ids.shape[0]
    if batch_videos.ndim != 5:
        raise AssertionError('The quantity of dimension for batch_videos must be 5 in for GANomaly 3D model')
            
    normal_path, abnormal_path = get_train_test_paths(folder_path, training)
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

def save_latent_vectors(batch_latents, batch_labels, batch_ids, batch_videos_ids, folder_path, training):
    """Function to save the batch of embeddings in the given folder path for train or test data, subdividing the normal and abnormal samples on different folders.
    Args:
        batch_latents (Array): A 2D numpy array with shape (b, z) where b = batch size and z = context vector size for the latent_vectors to be saved.
        batch_labels (Array): A 1D array with shape (b) for the labels of videos.
        batch_ids (Array): A 1D array with shape (b) for the patients ids of every video.
        batch_videos_ids (Array): A 1D array with shape (b) for the ids of videos in the batch.
        folder_path (String): The root path where the videos will be saved.
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_latents.shape[0] == batch_ids.shape[0] == batch_videos_ids.shape[0]
    if batch_latents.ndim != 2:
        raise AssertionError('The quantity of dimension for batch_latents must be 2 in for GANomaly 3D model')
            
    normal_path, abnormal_path = get_train_test_paths(folder_path, training)
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


def save_models(gen_model, disc_model, experiment_path, outputs_path, epoch = ""):
    """This function take the GANomaly 3D models and save them in the path given. Also it remove the previous models in the path together with the related outputs stored in outputs_folders to avoid saving a lot of innecessary files.
    Args:
        gen_model (Keras Model Instance): An instance of tf.keras.Model of the generator model to be saved.
        disc_model (Keras Model Instance): An instance of tf.keras.Model of the discriminator model to be saved.
        experiment_path (String): A string with the folder path in where the models will be saved.
        outputs_path (List[Str]): A list with the paths in which the model stores elements such as latent vectors, errors and videos.
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

    for path in outputs_path:
        os.system("rm -rf {}".format(os.path.join(path, "*")))

def save_model_results():