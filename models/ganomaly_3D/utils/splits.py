"""This file contains all the methods to split the data using kfolds or train, val and test dataset obtained from the normal and abnormal patients.
Version: 1.0
Made by: Edgar Rangel
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split

def __get_kfolds__(opts, videos_4_pat, normal_patients, normal_patients_ids, abnormal_patients, abnormal_patients_ids):
    """Functions that generate kfolds using the given parameters and returns the train, validation and test folds respectively.
    Args:
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        videos_4_pat (Dict): Dictionary of integers where the keys are the ids of the patients and the values are the total quantity of videos for each patient.
        normal_patients (List): List of tf.Data.Dataset elements in the same order as normal_patients_ids list containing the data for that patient.
        normal_patients_ids (List): List of integers containing the ids for normal patients.
        abnormal_patients (List): List of tf.Data.Dataset elements in the same order as abnormal_patients_ids list containing the data for that patient.
        abnormal_patients_ids (List): List of integers containing the ids for abnormal patients.
    """
    kfolds = opts["kfolds"]
    seed = opts["seed"]

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    train_folds = []
    val_folds = []
    test_folds = []
    val_totals = [0] * kfolds
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
            
        half = np.ceil(len(test_indexes) / 2).astype(np.int64)
        val_indexes = test_indexes[:half]
        data = normal_patients[val_indexes[0]]
        val_totals[k] += videos_4_pat[normal_patients_ids[val_indexes[0]]]
        for i in range(1, len(val_indexes)):
            data = data.concatenate(normal_patients[val_indexes[i]])
            val_totals[k] += videos_4_pat[normal_patients_ids[val_indexes[i]]]
        val_folds.append(data)

        final_indexes = test_indexes[half:]
        data = normal_patients[final_indexes[0]]
        test_totals[k] += videos_4_pat[normal_patients_ids[final_indexes[0]]]
        for i in range(1, len(final_indexes)):
            data = data.concatenate(normal_patients[final_indexes[i]])
            test_totals[k] += videos_4_pat[normal_patients_ids[final_indexes[i]]]
        test_folds.append(data)

        print("Kfold {}\n\tNormal train ids: {}\n\tNormal val ids: {}\n\tNormal test ids: {}".format(
            k + 1,
            [normal_patients_ids[i] for i in train_indexes],
            [normal_patients_ids[i] for i in val_indexes],
            [normal_patients_ids[i] for i in final_indexes]
        ))
    
    for k , (_, test_indexes) in enumerate(kf.split(abnormal_patients)):
        half = np.ceil(len(test_indexes) / 2).astype(np.int64)
        val_indexes = test_indexes[:half]

        data = abnormal_patients[val_indexes[0]]
        val_totals[k] += videos_4_pat[abnormal_patients_ids[val_indexes[0]]]
        for i in range(1, len(val_indexes)):
            data = data.concatenate(abnormal_patients[val_indexes[i]])
            val_totals[k] += videos_4_pat[abnormal_patients_ids[val_indexes[i]]]
        val_folds[k] = val_folds[k].concatenate(
            data
        ).shuffle(
            val_totals[k], 
            reshuffle_each_iteration=True
        ).batch(
            opts['batch_size']
        ).prefetch(-1)
        
        final_indexes = test_indexes[half:]
        data = abnormal_patients[final_indexes[0]]
        test_totals[k] += videos_4_pat[abnormal_patients_ids[final_indexes[0]]]
        for i in range(1, len(final_indexes)):
            data = data.concatenate(abnormal_patients[final_indexes[i]])
            test_totals[k] += videos_4_pat[abnormal_patients_ids[final_indexes[i]]]
        test_folds[k] = test_folds[k].concatenate(
            data
        ).shuffle(
            test_totals[k], 
            reshuffle_each_iteration=True
        ).batch(
            opts['batch_size']
        ).prefetch(-1)

        print("Kfold {}\n\tAbnormal val ids: {}\n\tAbnormal test ids: {}".format(
            k + 1,
            [abnormal_patients_ids[i] for i in val_indexes],
            [abnormal_patients_ids[i] for i in final_indexes]
        ))
    
    return train_folds, val_folds, test_folds

def __get_1_kfold__(opts, videos_4_pat, normal_patients, normal_patients_ids, abnormal_patients, abnormal_patients_ids):
    """Functions that generate 1 kfold (a.k.a as train, val and test partitions) using the given parameters and returns the train, validation and test folds respectively (a list with one element).
    Args:
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        videos_4_pat (Dict): Dictionary of integers where the keys are the ids of the patients and the values are the total quantity of videos for each patient.
        normal_patients (List): List of tf.Data.Dataset elements in the same order as normal_patients_ids list containing the data for that patient.
        normal_patients_ids (List): List of integers containing the ids for normal patients.
        abnormal_patients (List): List of tf.Data.Dataset elements in the same order as abnormal_patients_ids list containing the data for that patient.
        abnormal_patients_ids (List): List of integers containing the ids for abnormal patients.
    """
    total_test_size = opts["val_size"] + opts["test_size"]
    if opts["train_size"] + total_test_size != 1:
        raise AssertionError("The size for train, val and test partitions must sum in total 1.0. The sum give {}".format(opts["train_size"] + total_test_size))
    real_test_size = opts["test_size"] / total_test_size
    train_folds = []
    val_folds = []
    test_folds = []
    val_totals = 0
    test_totals = 0

    normal_train_indexes, temporal = train_test_split(range(len(normal_patients_ids)), test_size=total_test_size, random_state=opts["seed"])
    normal_val_indexes, normal_test_indexes = train_test_split(temporal, test_size=real_test_size, random_state=opts["seed"])

    print("Kfold {}\n\tNormal train ids: {}\n\tNormal val ids: {}\n\tNormal test ids: {}".format(
        opts["kfolds"],
        [normal_patients_ids[i] for i in normal_train_indexes],
        [normal_patients_ids[i] for i in normal_val_indexes],
        [normal_patients_ids[i] for i in normal_test_indexes]
    ))

    _, temporal = train_test_split(range(len(abnormal_patients_ids)), test_size=total_test_size, random_state=opts["seed"])
    abnormal_val_indexes, abnormal_test_indexes = train_test_split(temporal, test_size=real_test_size, random_state=opts["seed"])

    print("Kfold {}\n\tAbnormal val ids: {}\n\tAbnormal test ids: {}".format(
            opts["kfolds"],
            [abnormal_patients_ids[i] for i in abnormal_val_indexes],
            [abnormal_patients_ids[i] for i in abnormal_test_indexes]
        ))

    data = normal_patients[normal_train_indexes[0]]
    total_samples = videos_4_pat[normal_patients_ids[normal_train_indexes[0]]]
    for i in range(1, len(normal_train_indexes)):
        data = data.concatenate(normal_patients[normal_train_indexes[i]])
        total_samples += videos_4_pat[normal_patients_ids[normal_train_indexes[i]]]
    train_folds.append(
        data.shuffle(
            total_samples, 
            reshuffle_each_iteration=True
        ).batch(
            opts['batch_size']
        ).prefetch(-1)
    )

    data = normal_patients[normal_val_indexes[0]]
    val_totals += videos_4_pat[normal_patients_ids[normal_val_indexes[0]]]
    for i in range(1, len(normal_val_indexes)):
        data = data.concatenate(normal_patients[normal_val_indexes[i]])
        val_totals += videos_4_pat[normal_patients_ids[normal_val_indexes[i]]]

    for i in range(len(abnormal_val_indexes)):
        data = data.concatenate(abnormal_patients[abnormal_val_indexes[i]])
        val_totals += videos_4_pat[abnormal_patients_ids[abnormal_val_indexes[i]]]
    val_folds.append(
        data.shuffle(
            val_totals, 
            reshuffle_each_iteration=True
        ).batch(
            opts['batch_size']
        ).prefetch(-1)
    )

    data = normal_patients[normal_test_indexes[0]]
    test_totals += videos_4_pat[normal_patients_ids[normal_test_indexes[0]]]
    for i in range(1, len(normal_test_indexes)):
        data = data.concatenate(normal_patients[normal_test_indexes[i]])
        test_totals += videos_4_pat[normal_patients_ids[normal_test_indexes[i]]]

    for i in range(len(abnormal_test_indexes)):
        data = data.concatenate(abnormal_patients[abnormal_test_indexes[i]])
        test_totals += videos_4_pat[abnormal_patients_ids[abnormal_test_indexes[i]]]
    test_folds.append(
        data.shuffle(
            test_totals, 
            reshuffle_each_iteration=True
        ).batch(
            opts['batch_size']
        ).prefetch(-1)
    )

    return train_folds, val_folds, test_folds


def split_patients(opts, videos_4_pat, normal_patients, normal_patients_ids, abnormal_patients, abnormal_patients_ids):
    """Wrapper function to be called in the Ganomaly 3D script and select between apply kfolds or train_test_split method to separate the data.
    Args:
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        videos_4_pat (Dict): Dictionary of integers where the keys are the ids of the patients and the values are the total quantity of videos for each patient.
        normal_patients (List): List of tf.Data.Dataset elements in the same order as normal_patients_ids list containing the data for that patient.
        normal_patients_ids (List): List of integers containing the ids for normal patients.
        abnormal_patients (List): List of tf.Data.Dataset elements in the same order as abnormal_patients_ids list containing the data for that patient.
        abnormal_patients_ids (List): List of integers containing the ids for abnormal patients.
    """
    if opts["kfolds"] == 1:
        return __get_1_kfold__(opts, videos_4_pat, normal_patients, normal_patients_ids, abnormal_patients, abnormal_patients_ids)
    else:
        return __get_kfolds__(opts, videos_4_pat, normal_patients, normal_patients_ids, abnormal_patients, abnormal_patients_ids)