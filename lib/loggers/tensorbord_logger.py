from model.utils.logger import Logger #TODO refactor this

from loggers.logger_api import Logger


class TensorBoardLogger(Logger):
    def __init__(self, output_path):
        super(TensorBoardLogger, self).__init__(output_path)
        self.tensorboard = Logger(self.output_path)

    def scalar_summary(self, name, value, step):
        self.tensorboard.scalar_summary(name, value, step)
