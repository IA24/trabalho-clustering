import numpy as np
from constants import MIN_LIMIT_AGE, MIN_LIMIT_HEIGHT, MAX_LIMIT_AGE, MAX_LIMIT_HEIGHT



class Client:
    def __init__(self):
        self.age = self.__generate_age()
        self.height = self.__generate_height()
    
    @staticmethod
    def __generate_age():
        return np.random.uniform(MIN_LIMIT_AGE, MAX_LIMIT_AGE)

    @staticmethod
    def __generate_height():
        return np.random.uniform(MIN_LIMIT_HEIGHT, MAX_LIMIT_HEIGHT)
    