from model.feature_extractors.resnet_for_faster_rcnn import ResNetForFasterRCNN
from model.feature_extractors.vgg16_for_faster_rcnn import VGG16ForFasterRCNN


class FeatureExtractorsFactory(object):
    def __init__(self, net, pretrained_model_path=None):
        if net == 'vgg16':
            self.feature_extractors = VGG16ForFasterRCNN()
        elif net == 'res101':
            self.feature_extractors = ResNetForFasterRCNN()
        else:
            self.feature_extractors = None
            raise ValueError("Network undefined in config file, or the defined network is not supported.")

        if pretrained_model_path:
            self.feature_extractors.load_from_ckpt(pretrained_model_path)
        else:
            self.feature_extractors.init_params()
