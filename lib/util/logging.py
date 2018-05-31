import logging


def set_root_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file = logging.FileHandler(filename=log_path, mode='a')

    console = logging.StreamHandler()

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                  datefmt=None,
                                  style='%')
    file.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.addHandler(file)
    logger.addHandler(console)
