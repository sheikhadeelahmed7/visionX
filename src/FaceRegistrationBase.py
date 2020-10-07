from abc import ABC, abstractmethod


class FaceRegistrarBase(ABC):

    @abstractmethod
    def fileTypeCheck(self):
        """check basic file format e.g size, format etc"""
        pass
