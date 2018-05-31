import logging


def set_root_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file = logging.FileHandler(filename=log_path, mode='a')
    formatter_file = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s\n%(message)s',
                                       datefmt=None,
                                       style='%')
    file.setFormatter(formatter_file)

    console = logging.StreamHandler()
    formatter_console = logging.Formatter(fmt='%(message)s',
                                          datefmt=None,
                                          style='%')
    console.setFormatter(formatter_console)
    logger.addHandler(file)
    logger.addHandler(console)
