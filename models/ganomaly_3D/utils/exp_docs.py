"""This file contains the experiment documentation functions for GANomaly 3D model.
Version: 1.3
Made by: Edgar Rangel
"""

import os
from ....utils.common import format_index, get_next_last_item

def experiment_folder_path(base_path, isize, nc):
    """Function that generate the experiment folder given the base path and return the experiment folder path and experiment id.
    Args:
        base_path (String): The folder path that will contains several experiments on it organized by id.
        isize (Integer): Input size of the models.
        nc (Integer): Quantity of channels of models input.
    """
    experiment_id = get_next_last_item(base_path)
    experiment_path = os.path.join(base_path,
        "{id}_Ganomaly3D-{h}x{h}x{h}x{c}".format(
            id = format_index(experiment_id),
            h = isize,
            c = nc
        )
    )
    
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    return experiment_path, experiment_id
    
def get_metrics_path(experiment_path):
    """Function that generate the metrics experiment folder given the experiment path and return the metrics folder path.
    Args:
        experiment_path (String): The folder path of the experiment.
    """
    metric_save_path = os.path.join(experiment_path, "metrics")
    os.mkdir(metric_save_path)
    return metric_save_path

def get_outputs_path(experiment_path):
    """Function that generate the outputs experiment folder given the experiment path and return a vector with the folders path for input latent vectors, output latent vectors, real samples and fake samples generated respectively in that order.
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

def save_readme(save_path, opts, helptext_template, experiment_id, kfold):
    """This function format and complete the elements in the helptext template. This template must contain the variables listed in this function to be filled. Otherwise will throw an error and the method doesn't return anything.
    Args:
        save_path (String): A string in with the folder path where it will be saved.
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        helptext_template (String): A string containing the help text which will be saved along the experiment elements.
        experiment_id (String): A string with the format xxxx containing an unique number to identify the experiment.
        kfold (Integer): An integer indicating in which kfold the loop is executed.
    """
    with open(os.path.join(save_path, "README.txt"), "w+") as readme:
        readme.write(helptext_template.format(
            i = experiment_id,
            seed = opts["seed"],
            batch = opts["batch_size"],
            lr = opts["lr"],
            beta_1 = opts["beta_1"],
            beta_2 = opts["beta_2s"],
            epochs = opts["epochs"],
            nz = opts["nz"],
            nc = opts["nc"],
            ngf = opts["ngf"],
            extra_layers = opts["extra_layers"],
            w_gen = opts["w_gen"],
            size = opts["isize"],
            k = kfold
        ))
