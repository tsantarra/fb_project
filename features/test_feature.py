from collections import deque, Counter
from features.distribution import Distribution


class TestFeature:

    def __init__(self, audio_video_pairs):
        self.sources = list(audio_video_pairs)

    def weight_sources(self):
        return Distribution({source_pair: 0 for source_pair in self.sources})

    def update(self):
        return


