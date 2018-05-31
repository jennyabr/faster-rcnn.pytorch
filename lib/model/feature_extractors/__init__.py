from .faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors
from .resnet_for_faster_rcnn import ResNetForFasterRCNN
from .vgg_for_faster_rcnn import VGGForFasterRCNN

del faster_rcnn_feature_extractors
del vgg_for_faster_rcnn
del resnet_for_faster_rcnn

feature_extractors_classes = {cls.__name__: cls.__module__ for cls in FasterRCNNFeatureExtractors.__subclasses__()}
