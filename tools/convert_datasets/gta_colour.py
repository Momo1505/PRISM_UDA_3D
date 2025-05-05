# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def convert_to_train_id(file,is_pseudo=False):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = {
        0: 0, 11: 1, 20: 2, 30: 3, 32: 4, 35: 5, 60: 6, 64: 7, 70: 8, 80: 9,
        100: 10, 102: 11, 107: 12, 119: 13, 128: 14, 130: 15, 142: 16, 152: 17,
        153: 18, 156: 19, 170: 20, 180: 21, 190: 22, 220: 23, 230: 24, 232: 25,
        244: 26, 250: 27, 251: 28, 255: 29
    } if is_pseudo else {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    } 
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_suffix = '_pseudoTrainIds.png' if is_pseudo else '_labelTrainIds.png'
    new_file = file.replace('.png', new_suffix)
    assert file != new_file
    sample_class_stats['file'] = new_file
    if is_pseudo:
        image_2d = label_copy[:, :, :3].mean(axis=2).astype(np.uint8)
        Image.fromarray(image_2d, mode='L').save(new_file)
    else :
        Image.fromarray(label_copy).save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('--pseudo-dir', default='sam_colour', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def convert_to_train_id_with_flag(file_and_flag):
    """Wrapper function to unpack arguments and call convert_to_train_id."""
    file, is_pseudo = file_and_flag
    return convert_to_train_id(file, is_pseudo=is_pseudo)


def main():
    args = parse_args()
    gta_path = args.gta_path
    out_dir = args.out_dir if args.out_dir else gta_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(gta_path, args.gt_dir)
    pseudo_dir = osp.join(gta_path, args.pseudo_dir)

    # Collect files from labels
    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append((poly_file, False))  # False: not pseudo

    # Collect files from pseudo-labels
    for pseudo in mmcv.scandir(
            pseudo_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        pseudo_file = osp.join(pseudo_dir, pseudo)
        poly_files.append((pseudo_file, True))  # True: is pseudo

    poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            # Use the top-level function instead of lambda
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id_with_flag, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(
                convert_to_train_id_with_flag, poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    #if "pseudo" not in sample_class_stats["file"] :
    sample_class_stats_sorted = []
    for sample in sample_class_stats :
        if "pseudo" not in sample["file"] :
            sample_class_stats_sorted.append(sample)
    save_class_stats(out_dir, sample_class_stats_sorted)


if __name__ == '__main__':
    main()
