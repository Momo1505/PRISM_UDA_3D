# PRISM-UDA-3D

## Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/prism-uda
source ~/venv/prism-uda/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## Adding a new experiment 

To create a new experiment (for instance, on a new datasets), you will have to create two new file : test
- a config file, following the example that you can find in ```configs/mic/sample_config.py```and by correctly replacing ```NAME_OF_DATASET_FILE.py```, ```EXPERIMENT_NAME```and ```DATASET_NAME```.
- a dataset file, following the example that you can find in ```configs/_base_/datasets/sample_datasets.py``` and by correcly replacing ```ROOT_TO_SOURCE_DATASET``` and ```ROOT_TO_TARGET_DATASET```, know that only ``.tif`` files are currently suported for 3D data. If you want to use any other file extension, adapt the ``I3toLW4Dataset`` in ``mmseg/datasets/gta.py``.
For reminder, MMSEG is a framework originally created for 2D image segmentation, so for volume data, I needed to adapt some preprocessing. If you want to know what changes I made, look into any other files that does not have ``3D`` in their name, so that when you launch the training (presented below), and you encouter bugs,  you will know where to look.

## Training a new model
If you have read the PRISM UDA article, you should know that there are many components that make this method works:
- Daformer (changed to UNETR model, implementation can be found in ``mmseg/models/segmentors/encoder_decoder_3D.py``)
- Mixed training ( adapted to 3D)
- HRDA (need to be adapted)
- MIC (adapted to 3D)
- Refinement module (to be adapted)
First of all, try to run the code I left,  be careful to properly activate the venv with the following command :
```source ~/venv/prism-uda/bin/activate```

Once you have created the correct config and dataset file, you can run a new experiment by using 
```python run_experiments.py --config configs/mic/config_file.py```
The training scripts create a working directory in ```work_dirs/local-basic```

## Infering after training
In order to predict on the target domain after training, please use the following shell script :
```shell
sh test.sh work_dirs/local-basic/run_name/
```
For volume data, the inference still need to be adapted.
## Getting the results after inference 
```
python get_results.py --pred_path work_dirs/local-basic/run_name/preds/ --gt_path path-to-labels-ground-truth
```
Works for image data, but not for volume data

# What you need to do ...

## Run an experiment 
Before that, link the data directory to your project directory with ``ln -s /data2/sow/data`` (if ``sow`` was not deleted), since the data won't fit in your ssd with a copy.

Start by restarting the WeiH->I3 and LW4->I3 experiments in 2D (from the other repository) to get to grips with training, inference, and the code that outputs the results.
WARNING: In this case, the configuration and dataset files (in the configs folder) already exist, so it's up to you to find the right ones (>_<) ...

Restart the I3->LW4 training in 3D to verify that it works without bugs.
If everything goes well in 3D, you can finish adapting the remaining code for PRISM UDA in 3D, as described above. 
You can find the core of PRISM UDA method in ```mmseg/models/uda/dacs.py```. This file contains all the modules, including the refining network  which is trained in ``train_refinement_source`` method in the same file.
## Needed preprocessing for a new a new dataset

**First step** : You will need to apply SAM in an unserpervised way. I was not able to do it for volume, but for image, this is what I did:

I Pull this repo [SAM](https://github.com/facebookresearch/segment-anything) and install it. Copy the ``seg_anything_biomed.ipynb`` into the ``SAM`` project and run ``process_all`` function onto my new dataset.

An advice, even if it going to be slow, try, if possible, to apply this ``process_all`` function on high resolution image, like image of shape (2024,2024), because the SAM model might not work well on low resolution image (256,256).

I then Put the results in a subfolder called “sam” in the same way as the other in ``data`` directory. Just look at the structure of the files and folders to understand how it works.

**Deuxième étape** : Adapt the pre-processing file available in tools/convert_datasets. If you have volume data with more than 2 classes (0,255), you need to adapt the file  `I3-LW4_sam_3D.py`, you can use ``tools/convert_datasets/gta.py`` for inspiration. 
if your data has only two classes, do not modify  `I3-LW4_sam_3D.py` .

This is a data pre-processing code that is essential for the training code to work properly.
Once thes file has been adapted, it must be run on your new dataset using the following commands:
```bash
python tools/convert_datasets/{your preprocessing file}.py path_to_data_directory --nproc 8
```

You can now start your training.

## Acknowledgements

PRISM-UDA is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
