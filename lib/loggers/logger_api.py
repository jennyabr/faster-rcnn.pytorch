from abc import ABC, abstractmethod


class Logger(ABC):
    def __init__(self, output_path):
        self.output_path = output_path

    @abstractmethod
    def scalar_summary(self, name, value, step):
        raise NotImplementedError
