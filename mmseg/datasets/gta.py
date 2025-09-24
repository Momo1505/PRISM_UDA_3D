# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GTADataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, crop_pseudo_margins=None, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(GTADataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            pseudo_label_suffix="_pseudoTrainIds.png", #<------- ADDED
            split=None,
            **kwargs)
@DATASETS.register_module()
class I3toLW4Dataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, crop_pseudo_margins=None, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(I3toLW4Dataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='_labelTrainIds.tif',
            pseudo_label_suffix="_pseudoTrainIds.tif", #<------- ADDED
            split=None,
            **kwargs)
