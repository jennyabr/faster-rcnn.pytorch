# Faster R-CNN with Pytorch
This repository was built to create the most easily readable and extensible implementation of
Faster R-CNN in PyTorch. **The objective was to make it easier than ever for developers and
researchers to train\infer Faster R-CNN on your own data or extend it to similar algorithms
(e.g. Mask R-CNN) with just several lines of code** rather than duplicating thousands of existing
lines and re-combining them each time.

It was built on top of a fork from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) - the
most popular implementation of Faster R-CNN in PyTorch - an impressive and thorough implementation which
intended to create the first pure PyTorch repo for Faster R-CNN (in turn it built on top of several works of others,
which you can see in its README).
The level of abstraction which was in the focus of this work in order to reach the
objective stated above, was that of **Faster R-CNN as a meta architecture**, rather than the code that is
relevant only to its components (e.g. RPN, roi-pooling, etc.).

Re-design of components was performed only if the previous implementation made it hard to “use them”
as a Faster R-CNN user (e.g. if in order to choose which pooling method I use out of the implemented methods,
I need to do something that is more than a single line of code in my training script).

Highlighting some interesting points of this work can be found [here](https://docs.google.com/presentation/d/140-4eReJFky5E5YxZm6_9Fq1Mt74ZstWD25Va5P6QU0/edit?usp=sharing).


## Motivation for this work
“Writing code is rarely just a private affair between the developer and the computer.
**Code is not just meant for machines; it has human users. It is meant to be read by people, used by other developers,
maintained and built upon**.” While working on this project I have stumbled upon
[this very inspiring blog post](https://blog.keras.io/author/francois-chollet.html) written by Francois Chollet
the founder of Keras. I feel that these lines reflect one of the biggest pains in deep learning today: **GitHub is
full with brilliant open source libraries implementing complex papers, while the API UX design remains left behind,
thus making it hard for widespread adoption and use.**


## prerequisites
- Python 3 (I used 3.5.2)
- CUDA 8.0 or higher (I used CUDA 9.0)


## Preparation
1. Clone the repo:
```
git clone https://github.com/jennyabr/pytorch_faster_rcnn.git
```

2. Add the “lib” directory to the `PYTHONPATH`:
```
export PYTHONPATH=*path_to_repo*/lib
```

3. Prepare python virtual environment:
```
sudo pip install virtualenv                      # This may already be installed
virtualenv -p python3 [path_to_environment].env  # Create a virtual environment (python3)
                                                 # Note: you can also use "virtualenv .env"
                                                 # to use your default python (usually python 2.7)
source [path_to_environment].env/bin/activate    # Activate the virtual environment
pip install -r lib/requirements.txt              # Install dependencies
```
* To exit the virtual environment run: `deactivate`

4. compile (from lib directory):
```
cd lib
bash make.sh
```
* You will be asked for your GPU architecture,
you can find it [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
(if it is SM_60, enter 60 when you’re prompted).


## Data
The original project supports PASCAL_VOC 07+12, COCO and Visual Genome.
I worked only with PASCAL_VOC (testing on other dataset was added to the TODO list).
These are the data preperation instruction as were provided in the original repo:
- PASCAL_VOC 07+12: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
to prepare VOC datasets.
- COCO: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
to prepare the data.
- Visual Genome: Please follow the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
to prepare Visual Genome dataset. You need to download the images and object annotation files first,
and then perform preprocessing to obtain the vocabulary and cleansed annotations based on the
scripts provided in this repository.


## Train and Inference
In the demos directory you will find several scripts that demonstrate how to use the library:
All the scripts start with setting the config and the logger.
1. Create the config file for your run:
1.a. Use one of the existing config files (e.g. vgg16.yml) or create your own.
1.b. The config file defines all the desired hyper parameters for the run, as well as the experiment name,
paths for saving the checkpoints, predictions, etc..
1.c. The existing config files define only a small subset of the possible hyper-parameters and use the default
values of many others. You can see the full list of hyper-parameters you can define, and their default values,
in [defaults.yml](lib/utils/defaults.yml)
2. In order to run end to end (train and inference) run one of the existing demo scripts
(e.g. [vgg_and_resnet_e2e.py](https://github.com/jennyabr/pytorch_faster_rcnn/blob/master/demos/vgg_and_resnet_e2e.py))
or a copy of them with the path to your config file:
```
python vgg_and_resnet_e2e.py
```


## TODOs
- [ ] Compare results to the original implementation while running on multiple GPUs.
- [ ] Test on COCO, Visual Genome and VOC-2012.
- [ ] Upgrade to torch 0.4.
- [ ] Optimize metric averaging to occur on GPU (some interesting points to test are listed [here](https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/)).
- [ ] Fix incorrect usage of `torch.FloatTensor(NUMBER)`.
- [ ] Number of output coords in faster rcnn should be a parameter (currently hardcoded to 4).
- [ ] Reloading faster rcnn from checkpoint should enable manually overriding `num_classes` **and performing random initialization on the X last layers for fine-tuning**.
- [ ] Make the `load_session_from_ckpt` function in `ckpt_utils.py` independent from FasterRCNN (i.e. don't use the `FasterRCNN` constructor)
- [ ] `ConfigProvider` should enable setting attributes inside `_cfg` for h.p. sweeps.
- [ ] Config should be divided into logical units.
- [ ] Method `_freeze_layers` in `FeatureExtractorDuo` should be a recursion.
- [ ] Check if the loss can be removed from the state (i.e. remove from `self`) of the modules.
- [ ] **Add tests**.
- [ ] Implement Mask RCNN.

# Contributing
I want to make contributing to this project as easy and transparent as possible.

### Pull Requests
I welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. Make sure your code lints.
4. Ensure no regressions in baseline model speed and accuracy (all scripts in the demos directory).

### Issues
GitHub issues will be largely unattended and are mainly intended as a community
forum for collectively debugging issues, hopefully leading to pull requests with
fixes when appropriate.

### Coding Style
* 4 spaces for indentation rather than tabs
* 120 character line length
* PEP8 formatting

## License
This project is licensed under the terms of the [MIT license](LICENSE.md).
