from collections import deque, Counter
import cv2
from features.distribution import Distribution


class VideoMovementFeature:

    def __init__(self, audio_video_pairs, window_length=10, thrash_limit=3):
        self.window_length = window_length
        self.thrash_limit = thrash_limit
        self.sources = list(audio_video_pairs)
        self.window = deque()
        self.last_selected = None
        self.time_since_switch = 0
        self.last_frames = []

    def weight_sources(self):
        # Examination of sliding window
        source_count = Counter(self.window)
        max_choice = source_count.most_common(1)[0][0]  # returns list of pairs ala [(item, count)]

        # Consideration for thrashing
        if max_choice != self.last_selected:
            if self.time_since_switch > self.thrash_limit or self.last_selected is None:
                    self.last_selected = max_choice
                    self.time_since_switch = 0

        # Vote distribution
        vote = Distribution({source_pair: 0 for source_pair in self.sources})
        vote[self.last_selected] = 1.0
        return vote

    def update(self):
        # Initial conditions
        if not self.last_frames:
            self.last_frames = [video.read() for audio, video in self.sources]
            self.window.append(self.sources[0])
            return

        # Progress tracking vars
        if len(self.window) > self.window_length:
            self.window.popleft()
        self.time_since_switch += 1

        # Process new frames
        new_frames = [video.read() for audio, video in self.sources]
        diffs = [cv2.absdiff(new, old) for new, old in zip(new_frames, self.last_frames)]
        diffs = [cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY) for frame in diffs]
        diffs = [frame.sum() for _, frame in diffs]
        max_index = diffs.index(max(diffs))

        # Update vars
        self.last_frames = new_frames
        self.window.append(self.sources[max_index])


