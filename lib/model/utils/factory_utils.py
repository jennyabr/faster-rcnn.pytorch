from importlib import import_module


def get_class_from_package(package_full_path, class_rel_path, abstract_class):
    module_name, class_name = class_rel_path.rsplit('.', 1)
    try:
        class_module = import_module(package_full_path + '.' + module_name)

        returned_class = getattr(class_module, class_name)
        if not issubclass(returned_class, abstract_class):
            raise ImportError(
                "{} is not a subclass of the given abstract class.".format(returned_class))

        return returned_class
    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of the package!'.format(class_name))


def get_optimizer_class(optimizer_name):
    from torch.optim import Optimizer
    module_name = optimizer_name.lower()
    class_rel_path = module_name + '.' + optimizer_name
    optimizer_class = get_class_from_package('torch.optim', class_rel_path, Optimizer)
    return optimizer_class
