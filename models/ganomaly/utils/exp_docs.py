"""This file contains the experiment documentation functions for GANomaly nets.
Version: 1.1
Made by: Edgar Rangel
"""

import os

def experiment_folder_path(base_path, model_dimension, isize, nc):
    """Function that generate the experiment folder given the base path and return the experiment 
    folder path and experiment id.
    Args:
        base_path (String): The folder path that will contains several experiments on it organized by id.
        model_dimension (String): Dimensionality on which the model's convolutions operate, can be "3D" or "2D".
        isize (Integer): Input size of the models.
        nc (Integer): Quantity of channels of models input.
    """
    assert model_dimension in ["2D", "3D"]

    experiment_id = 1
    experiments = sorted(os.listdir(base_path))
    if len(experiments) == 0:
        experiment_id = '000' + str(experiment_id)
    else:
        experiment_id = int(experiments[-1].split("_")[0]) + 1
        if experiment_id < 10:
            experiment_id = '000' + str(experiment_id)
        elif experiment_id < 100:
            experiment_id = '00' + str(experiment_id)
        elif experiment_id < 1000:
            experiment_id = '0' + str(experiment_id)
        else:
            experiment_id = str(experiment_id)

    experiment_path = os.path.join(base_path,
        "{id}_Ganomaly_{d}-".format(
            id = experiment_id,
            d = model_dimension
        )
    )

    if model_dimension == "2D":
        experiment_path += "{h}x{h}x{c}".format(
            h = isize,
            c = nc
        )
    else:
        experiment_path += "{h}x{h}x{h}x{c}".format(
            h = isize,
            c = nc
        )
    
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    return experiment_path, experiment_id
    
def get_metrics_path(experiment_path):
    """Function that generate the metrics experiment folder given the experiment path and return 
    the metrics folder path.
    Args:
        experiment_path (String): The folder path of the experiment.
    """
    metric_save_path = os.path.join(experiment_path, "metrics")
    os.mkdir(metric_save_path)
    return metric_save_path

def get_outputs_path(experiment_path):
    """Function that generate the outputs experiment folder given the experiment path and return 
    a vector with the folders path for input latent vectors, output latent vectors, real samples 
    and fake samples generated respectively in that order.
    Args:
        experiment_path (String): The folder path of the experiment.
    """
    output_path = os.path.join(experiment_path, "outputs")
    os.mkdir(output_path)
    paths = [
        {"folder":"latent_vectors", "subfolders": ["input_generator", "output_generator", "input_discriminator", "output_discriminator"]}, 
        {"folder":"samples", "subfolders":["real", "fake", "substraction"]},
        {"folder":"errors", "subfolders":["adversarial", "contextual", "encoder"]},
        {"folder": "graphics", "subfolders":["quantitative", "qualitative", "visuals"]}
    ]
    final_paths = []
    for path in paths:
        os.mkdir(os.path.join(output_path, path["folder"]))
        for folder in path["subfolders"]:
            final_path = os.path.join(output_path, path["folder"], folder)
            os.mkdir(final_path)
            final_paths.append(final_path)

    return final_paths
