class DataPool:
    def __init__(self, initial_data: list[dict]):
        self.data = initial_data

    def add_samples(self, samples: list[dict]):
        self.data.extend(samples)

    def remove_samples(self, samples: list[dict]):
        videos_to_remove = {s["video"] for s in samples}
        self.data = [d for d in self.data if d["video"] not in videos_to_remove]

    def size(self) -> int:
        return len(self.data)
