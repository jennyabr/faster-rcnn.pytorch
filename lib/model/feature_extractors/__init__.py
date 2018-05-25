
from .faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors
from .vgg_for_faster_rcnn import VGGForFasterRCNN
from .resnet_for_faster_rcnn import ResNetForFasterRCNN

del faster_rcnn_feature_extractors
del vgg_for_faster_rcnn
del resnet_for_faster_rcnn



feature_extractors_classes = {cls.__name__: cls.__module__ for cls in FasterRCNNFeatureExtractors.__subclasses__()}

# TODO: delete this
#print("---1--- {}".format(__package__))

# import os
# import importlib
# import pkgutil
#
# pkg_dir = os.path.dirname(__file__)
# for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
#     importlib.import_module('.' + name, __package__)


#dir(torch.optim)
# from .adadelta import Adadelta
# from .adagrad import Adagrad
# from .adam import Adam
# from .sparse_adam import SparseAdam
# from .adamax import Adamax
# from .asgd import ASGD
# from .sgd import SGD
# from .rprop import Rprop
# from .rmsprop import RMSprop
# from .optimizer import Optimizer
# from .lbfgs import LBFGS
# from . import lr_scheduler
#
# del adadelta
# del adagrad
# del adam
# del adamax
# del asgd
# del sgd
# del rprop
# del rmsprop
# del optimizer
# del lbfgs