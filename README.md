# Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition

This repo contains the reference source code for our CVPR 2020 paper [Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition](https://arxiv.org/abs/2004.00705).

## Environment

Python 3.7

Pytorch 1.1.0 with CUDA 9.0

tensorboardX

## Set up dataset

In our experiments, we use four datasets: [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [NABirds](https://dl.allaboutbirds.org/nabirds), [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and [OID-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/oid/). 

You have two options to download this data:

- Manually download them using the hyper-links provided above, and then extract them into the `dataset` folder.
- If you don't want to download and extract them one by one, you can also go into the `dataset` folder and execute `download.sh`. Before doing that, though, please go to the official [NABirds](https://dl.allaboutbirds.org/nabirds) website and register using your name and email address, and also accept their terms of use.

After download is finished, navigate to the `dataset` folder and execute `init.sh` to generate the dataset we use for training and evaluation. More details about dataset split can be found in the paper.

## Train and test

For experiments on CUB and FGVC, each model has its own individual folder in the `CUB` or `FGVC` directory respectively. 

For traininng, you just need to go to the model folder you wish to run, and execute `Con4.sh` or `ResNet18.sh` for the 4-layer ConvNet or ResNet18 backbone. The hyper-parameters have already been set to the values given in the supplementary materials. We set the default gpu device to 0. If you want to specify others, just change the `--gpu` argument in the `.sh` script.

The training and validation accuracies are displayed in both the std output and the generated `*.log` file. The training history, including losses, train/validation accuracy and heatmap visualization, can also be displayed via tensorboard. The tensorboard summary is located in the `log_*` folder. During the training process, the model snapshot with the best validation performance will be saved in `model_*.pth`.

After training is complete, the script will automatically evaluate the final model on the corresponding test set, and output the test accuracy numbers in both the std output and `*.log` file.

## Train with less part annotation

For the ablation study on training on the CUB dataset with less part annotation, navigate to `proto+PN_less_annot` in the `CUB` directory. The `.sh` scripts have been set to training with 20% part annotation and batch size 7. If you want to train with other percentages of annotation, change both the  `--percent` and `--batch_size` arguments in the scripts as described in the supplementary matrials.


## Citation
If you find our code or paper useful, please consider citing our work using the following bibtex:
```
@inproceedings{tang2020revisiting,
  title={Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition},
  author={Tang, Luming and Wertheimer, Davis and Hariharan, Bharath},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14352--14361},
  year={2020}
}
```


## Updates
05/27/2021: As pointed out in this [issue](https://github.com/Tsingularity/PoseNorm_Fewshot/issues/2), the website for OID-Aircraft seems to be down for now. As an alternative for downloading the dataset, I upload one copy to this google drive [link](https://drive.google.com/file/d/10vKcoS6-JFEpioD_FStJZbWknFCfPOnU/view?usp=sharing). More details about the dataset could be found in its original [paper](https://www.robots.ox.ac.uk/~karen/pdf/vedaldi14understanding.pdf). If you wanna download the dataset from google drive via command line directly, please refer to this [line](https://github.com/Tsingularity/PoseNorm_Fewshot/blob/master/dataset/download.sh#L3) of code in `download.sh`, or you can refer to our recent [FRN](https://github.com/Tsingularity/FRN) repository for more details.
