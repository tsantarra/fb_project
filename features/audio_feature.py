from distribution import Distribution
from collections import deque, Counter


def reduce_audio_frames(frames):
    """ I believe the format is list of frames, each of which has a value per channel
    """
    return sum(sum(data)/len(data) for data in frames)/len(frames)


class AudioFeature:

    def __init__(self, audio_video_pairs, window_length=50, thrash_limit=25):
        self.window_length = window_length
        self.thrash_limit = thrash_limit
        self.sources = list(audio_video_pairs)
        self.window = deque()
        self.last_selected = None
        self.time_since_switch = 0

    def weight_sources(self):
        vote = Distribution({source_pair[1]: 0 for source_pair in self.sources})
        vote[self.last_selected[1]] = 1.0
        return vote

    def update(self):
        # Progress tracking vars
        if len(self.window) > self.window_length:
            self.window.popleft()
        self.time_since_switch += 1

        # Append max source
        loudest_source = max(self.sources, key=lambda av: reduce_audio_frames(av[0].read()))
        print(self.sources.index(loudest_source))
        self.window.append(loudest_source)

        # Examination of sliding window
        source_count = Counter(self.window)
        max_choice = source_count.most_common(1)[0][0]  # returns list of pairs ala [(item, count)]

        # Consideration for thrashing
        if max_choice != self.last_selected:
            if self.time_since_switch > self.thrash_limit or self.last_selected is None:
                    self.last_selected = max_choice
                    self.time_since_switch = 0

