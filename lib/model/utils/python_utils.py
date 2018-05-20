
from importlib import import_module

from model.feature_extractors.faster_rcnn_feature_extractors import FasterRCNNFeatureExtractors


def get_class_from_package(package_full_path, class_rel_path, abstract_class):
    try:
        module_name, class_name = class_rel_path.rsplit('.', 1)
        class_module = import_module(package_full_path + '.' + module_name)

        returned_class = getattr(class_module, class_name)
        if not issubclass(returned_class, abstract_class):
            raise ImportError(
                "{} is not a subclass of the given abstract class.".format(returned_class))

        return returned_class
    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of the package!'.format(class_name))


if __name__=='__main__':
    get_class_from_package('lib.model.feature_extractors', 'vgg16_for_faster_rcnn.VGG16ForFasterRCNN',
                           FasterRCNNFeatureExtractors)