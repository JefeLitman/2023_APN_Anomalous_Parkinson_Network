"""This file contain the method that execute the selected mode to run the GANomaly 3D model. This file contain specifically the joint of preprocessing, model modes (execution loops) and results obtained. Its important that the scripts files never import any function or method outside the mandatory method called run.
Version: 1.3
Made by: Edgar Rangel
"""

def run(mode):
    """Function that is similar to the notebook of Ganomaly 3D model, and it lets you run the model directly from the scripts, you only need to call this method in the run.py file specifying one mode to run the model.
    Args:
        mode (String): A string with the exact names (without .py) of files inside the folder modes in the different models, e.g. "train_eval".
    """
    print("Starting Ganomaly 3D model execution...\nLoading hiperparameters...")
    from models.ganomaly_3D.hiperparameters import get_options
    opts = get_options()

    print("Hiperparameters loaded!\nSetting th GPU to be used...")
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts["gpus"]

    print("GPU selected was {}\nLoading libraries and methods...".format(opts["gpus"]))
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import KFold

    from datasets.gait_v2.extraction_methods import get_data
    from models.ganomaly_3D.modes.eval import exec_loop as test
    from models.ganomaly_3D.modes.train import exec_loop as train
    from models.ganomaly_3D.modes.train_eval import exec_loop as train_eval
    from models.ganomaly_3D.data_preprocessing import preprocess_gait_dataset
    from utils.metrics import get_true_positives, get_true_negatives, get_false_positives, get_false_negatives, get_AUC, get_mean

    print("Libraries loaded!\nSetting memory constraint for GPU {}".format(opts["gpus"]))
    if os.getenv("CUDA_VISIBLE_DEVICES") != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.debugging.set_log_device_placement(False)

    print("GPU {} configured correctly!\nLoading raw data...".format(opts["gpus"]))
    total_data = get_data(opts["dataset_path"], opts["n_cpus"])
    
    print("Raw data loaded correctly!\nGetting basic data information...")
    shape_videos = []
    labels_videos = []
    patients_ids = []
    for x, y, z in total_data:
        shape_videos.append(x.numpy().shape)
        labels_videos.append(y.numpy())
        patients_ids.append(z.numpy())
    shape_videos = np.r_[shape_videos]
    labels_videos = np.r_[labels_videos]
    patients_ids = np.r_[patients_ids]
    print("Data information obtained about the data...")
    print("Total videos: ", shape_videos.shape[0])
    print("Min value of frames: ", np.min(shape_videos[:,0]))
    print("Max value of frames: ", np.max(shape_videos[:,0]))
    print("Mean value of frames: ", np.mean(shape_videos[:,0]))
    print("Unique ids: ", np.unique(patients_ids))

    print("Basic information displayed!\nCalculating total videos by class...")
    ns = {i:0 for i in np.unique(labels_videos)}
    videos_4_pat = {i:0 for i in np.unique(patients_ids)}
    for i, forma in enumerate(shape_videos):
        frames = opts["isize"]
        to_sum = np.ceil(forma[0] / frames).astype(np.int64)
        videos_4_pat[patients_ids[i]] += to_sum
        ns[labels_videos[i]] += to_sum
    for i in ns:
        print("Video clips for label {}: {}".format(i, ns[i]))

    print("Calculations finished!\nSeparating patients by classes...")
    normal_patients_ids = np.unique(patients_ids[labels_videos == opts['normal_class']])
    abnormal_patients_ids = np.unique(patients_ids[labels_videos != opts['normal_class']])
    print("Normal patients ids: {}".format(normal_patients_ids))
    print("Abnormal patients ids: {}".format(abnormal_patients_ids))
    
    print("Separation done!\nSetting data preprocessing pipeline...")
    normal_patients, abnormal_patients = preprocess_gait_dataset(
        total_data, 
        opts,
        normal_patients_ids,
        abnormal_patients_ids
    )

    print("Preprocessing finished!\nLoading the readme template for exps...")
    readme_template = opts['readme']

    print("Template loaded!\nCalculating the kfolds for the experiments...")
    kfolds = opts["kfolds"]
    seed = opts["seed"]

    print("Total kfolds to be used: {}\nSeed used to calculate the kfolds: {}".format(opts["kfolds"], opts["seed"]))
    # Data partition for train and test with kfold
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    train_folds = []
    test_folds = []
    test_totals = [0] * kfolds
    for k, (train_indexes, test_indexes) in enumerate(kf.split(normal_patients)):
        data = normal_patients[train_indexes[0]]
        total_samples = videos_4_pat[normal_patients_ids[train_indexes[0]]]
        for i in range(1, len(train_indexes)):
            data = data.concatenate(normal_patients[train_indexes[i]])
            total_samples += videos_4_pat[normal_patients_ids[train_indexes[i]]]
        train_folds.append(
            data.shuffle(
                total_samples, 
                reshuffle_each_iteration=True
            ).batch(
                opts['batch_size']
            ).prefetch(-1)
        )
            
        data = normal_patients[test_indexes[0]]
        test_totals[k] += videos_4_pat[normal_patients_ids[test_indexes[0]]]
        for i in range(1, len(test_indexes)):
            data = data.concatenate(normal_patients[test_indexes[i]])
            test_totals[k] += videos_4_pat[normal_patients_ids[test_indexes[i]]]
        test_folds.append(data)

        print("Kfold {}\n\tNormal train ids: {}\n\tNormal test ids: {}".format(
            k + 1,
            [normal_patients_ids[i] for i in train_indexes],
            [normal_patients_ids[i] for i in test_indexes]
        ))
    
    for k , (_, test_indexes) in enumerate(kf.split(abnormal_patients)):
        data = abnormal_patients[test_indexes[0]]
        test_totals[k] += videos_4_pat[abnormal_patients_ids[test_indexes[0]]]
        for i in range(1, len(test_indexes)):
            data = data.concatenate(abnormal_patients[test_indexes[i]])
            test_totals[k] += videos_4_pat[abnormal_patients_ids[test_indexes[i]]]
        test_folds[k] = test_folds[k].concatenate(
            data
        ).shuffle(
            test_totals[k], 
            reshuffle_each_iteration=True
        ).batch(
            opts['batch_size']
        ).prefetch(-1)

        print("Kfold {}\n\tAbnormal test ids: {}".format(
            k + 1,
            [abnormal_patients_ids[i] for i in test_indexes]
        ))
        

    print("Kfolds calculated!\nCreating metrics for the model...")
    TP = get_true_positives()
    TN = get_true_negatives()
    FP = get_false_positives()
    FN = get_false_negatives()
    gen_loss = get_mean()
    disc_loss = get_mean()
    AUC = get_AUC()

    print("Metrics created!\nStarting model execution...")
    for k in range(opts['kfolds']):
        if mode == "train_eval":
            train_eval(
                opts,
                readme_template,
                k + 1,
                TP,
                TN,
                FP,
                FN,
                AUC,
                gen_loss,
                disc_loss,
                train_folds[k],
                test_folds[k]
            )
        elif mode == "train":
            train(
                opts,
                readme_template,
                k + 1,
                TP,
                TN,
                FP,
                FN,
                AUC,
                gen_loss,
                disc_loss,
                train_folds[k]
            )
        elif mode == "eval":
            test(
                opts,
                TP,
                TN,
                FP,
                FN,
                AUC,
                train_folds[k],
                test_folds[k]
            )
        else:
            raise AssertionError('The mode {} is not available for the Ganomaly 3D modes.'.format(mode))

    print("Model finished!\nEnd of the report.")