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

    def remove_samples_from_list_of_video_names(self, video_names: list[str]):
        video_names_set = set(video_names)

        removed = [d for d in self.data if d["video"] in video_names_set]
        self.data = [d for d in self.data if d["video"] not in video_names_set]

        return removed
