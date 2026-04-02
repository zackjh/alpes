from abc import ABC, abstractmethod
import random
from datapool import DataPool


class Sampler(ABC):
    @abstractmethod
    def get_samples(self, n: int, data_pool: DataPool) -> list[dict]:
        pass


class RandomSampler(Sampler):
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def get_samples(self, n: int, data_pool: DataPool) -> list[dict]:
        data: list[dict] = data_pool.data

        n = min(n, len(data))

        samples = self.rng.sample(data, n)
        return samples
