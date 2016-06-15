from abc import ABCMeta, abstractmethod


class Feature(metaclass=ABCMeta):

    @abstractmethod
    def weight_sources(self):
        pass

    @abstractmethod
    def update(self):
        pass