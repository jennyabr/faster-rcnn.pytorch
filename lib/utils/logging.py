import logging


def set_root_logger(log_path):
    logging.shutdown()  # clearing up logging in case it was already initialized
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    formatter_console = logging.Formatter(fmt='%(message)s',
                                          datefmt=None,
                                          style='%')
    console.setFormatter(formatter_console)
    logger.addHandler(console)

    for h in logger.handlers:
        if type(h) == logging.FileHandler:
            logger.removeHandler(h)
    file = logging.FileHandler(filename=log_path, mode='a', encoding='utf-8')
    formatter_file = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s\n\t%(message)s\n',
                                       datefmt=None,
                                       style='%')
    file.setFormatter(formatter_file)
    logger.addHandler(file)
