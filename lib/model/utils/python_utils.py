
from importlib import import_module


def get_class_from_package(class_name, package_name, abstract_class):
    try:
        if '.' in class_name:
            module_name, class_name = class_name.rsplit('.', 1)
        else:
            module_name = class_name
            class_name = class_name.capitalize()

        class_module = import_module('.' + module_name, package=package_name)

        returned_class = getattr(class_module, class_name)
        if not issubclass(returned_class, abstract_class):
            raise ImportError(
                "{} is not a subclass of the given abstract class.".format(returned_class))

        return returned_class
    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of the package!'.format(class_name))


if __name__=='__main__':
    print(1)