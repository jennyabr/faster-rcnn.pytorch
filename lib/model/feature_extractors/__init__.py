from .faster_rcnn_feature_extractor_duo import FasterRCNNFeatureExtractorDuo
from .resnet_for_faster_rcnn import ResNetForFasterRCNN
from .vgg_for_faster_rcnn import VGGForFasterRCNN

del faster_rcnn_feature_extractor_duo
del vgg_for_faster_rcnn
del resnet_for_faster_rcnn

feature_extractors_duo_classes = {cls.__name__: cls.__module__ for cls in FasterRCNNFeatureExtractorDuo.__subclasses__()}
