from .feature_extractor_duo import FeatureExtractorDuo
from .resnet_feature_extractor_duo import ResNetFeatureExtractorDuo
from .vgg_feature_extractor_duo import VGGFeatureExtractorDuo

del feature_extractor_duo
del vgg_feature_extractor_duo
del resnet_feature_extractor_duo

feature_extractors_duo_classes = {cls.__name__: cls.__module__ for cls in FeatureExtractorDuo.__subclasses__()}
