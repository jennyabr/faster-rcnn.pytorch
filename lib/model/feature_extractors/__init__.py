
from .faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors
from .vgg16_for_faster_rcnn import VGG16ForFasterRCNN
from .resnet_for_faster_rcnn import ResNetForFasterRCNN

del faster_rcnn_feature_extractors
del vgg16_for_faster_rcnn
del resnet_for_faster_rcnn

#import inspect

#clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)




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