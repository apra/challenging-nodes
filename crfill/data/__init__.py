"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from util.metadata_utils import get_paths_and_nodules, get_paths_negatives, get_paths
import pickle
from pathlib import Path

def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.num_workers),
        drop_last=opt.isTrain
    )
    return dataloader


def create_dataloader_trainval(opt):
    # get the path to images and the nodules locations, these are already shuffled
    if opt.model == 'arrange':
        paths_and_nodules = get_paths_and_nodules(opt.train_image_dir, opt.include_chexpert,
                                              opt.include_mimic, opt.node21_resample_count)
    elif opt.model == 'arrangedoubledisc' or opt.model == 'arrangeskipconn' or opt.model == 'arrangeplacelesion':
        paths_positive = get_paths_and_nodules(opt.train_image_dir, opt.include_chexpert,
                                              opt.include_mimic, opt.node21_resample_count)
        paths_negative = get_paths_negatives(opt.train_image_dir)
        paths_lesions = get_paths(opt.train_lesion_dir)
        metadata_file = Path(opt.train_lesion_dir).parent / Path('metadata.pkl')
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        paths_and_nodules = [paths_positive, paths_negative, paths_lesions]
    elif opt.model == 'vae':
        paths_and_nodules = get_paths(opt.train_image_dir)
        metadata_file = Path(opt.train_image_dir).parent / Path('metadata.pkl')
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
    else:
        paths_and_nodules = get_paths(opt.train_image_dir)

    dataset = find_dataset_using_name(opt.dataset_mode_train)
    instance = dataset()
    print(dataset)
    if opt.model == 'vae' or opt.model == 'arrangeplacelesion':
        instance.initialize(opt,paths_and_nodules, 'train', metadata)
    else:
        instance.initialize(opt, paths_and_nodules, 'train')
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    print(f"Num workers: {int(opt.num_workers)}. Threads available: {torch.get_num_threads()}")
    dataloader_train = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.num_workers),
        drop_last=True
    )
    dataset = find_dataset_using_name(opt.dataset_mode_train)
    instance = dataset()
    if opt.model == 'vae' or opt.model == 'arrangeplacelesion':
        instance.initialize(opt, paths_and_nodules, 'valid', metadata)
    else:
        instance.initialize(opt, paths_and_nodules, 'valid')
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader_val = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.num_workers),
        drop_last=True
    )
    return dataloader_train, dataloader_val
