from collections import deque, Counter
import cv2
from features.distribution import Distribution
from pipeline_interfaces import PipelineFunction

class VideoMovementFeature(PipelineFunction):

    def __init__(self, video_sources, window_length=10, thrash_limit=3):
        self.window_length = window_length
        self.thrash_limit = thrash_limit
        self.sources = list(video_sources)
        self.window = deque()
        self.last_selected = None
        self.time_since_switch = 0
        self.last_frames = []

    def start(self):
        pass

    def update(self):
        # Initial conditions
        if not self.last_frames:
            self.last_frames = {source: source.read() for source in self.sources}
            self.window.append(self.sources[0])
            return

        # Progress tracking vars
        if len(self.window) > self.window_length:
            self.window.popleft()
        self.time_since_switch += 1

        # Process new frames
        new_frames = {source: source.read() for source in self.sources}

        diffs = {source: cv2.absdiff(new_frames[source], self.last_frames[source]) for source in new_frames
                 if (new_frames[source] is not None and self.last_frames[source] is not None)}

        diffs = {source: cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] for source, diff in diffs.items()}
        diffs = {source: frame.sum() for source, frame in diffs.items()}
        max_source = max(diffs, key=lambda source: diffs[source], default=self.sources[0])

        # Update vars
        self.window.append(max_source)
        self.last_frames = new_frames

    def read(self):
        pass

    def close(self):
        pass


    def complete(self):
        return True  #self.process.exitcode is not None

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
        vote = Distribution({source: 0 for source in self.sources})
        vote[self.last_selected] = 1.0
        return vote