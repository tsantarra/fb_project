from collections import deque, Counter

import cv2

from util.distribution import Distribution
from util.pipeline import PipelineProcess, get_all_from_queue
from util.schedule import create_periodic_event


class VideoMovementFeature(PipelineProcess):
    """
    Votes for a video stream based on which stream has the most pairwise frame differences within a sliding window.
    """

    def __init__(self, feature_id, video_sources, window_length=10):
        super().__init__(pipeline_id=feature_id,
                         target_function=VideoMovementFeature.establish_process_loop,
                         params=(window_length, [source.id for source in video_sources]),
                         sources=video_sources)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, window_length, source_ids):
        import numpy
        window = deque(maxlen=window_length)  # A sliding window containing the most active stream for each frame
        last_frames = {source_id: numpy.zeros((480, 640, 3), dtype='uint8') for source_id in source_ids}
        width, height = 640, 480

        def weight_sources():
            nonlocal window, last_frames

            # We're going to collect new frames by rolling through all awaiting updates, saving only the last actual
            # frame for each source. In effect, to avoid computational slowdown, we're diff-ing only with the most
            # recent frames.
            new_frames = {}
            for update_step in get_all_from_queue(input_queue):
                for source_id, frame_list in update_step.items():
                    if frame_list:
                        frame = frame_list[-1]
                        new_frames[source_id] = frame if frame.shape == (height, width, 3) \
                            else cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            for source in last_frames.keys() - new_frames.keys():
                new_frames[source] = last_frames[source]

            # Calculated diffs between new and last frames
            diffs = {source: cv2.absdiff(new_frames[source], last_frames[source]) for source in new_frames
                     if (new_frames[source] is not None and last_frames[source] is not None)}
            diffs = {source: cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] for source, diff in diffs.items()}
            diffs = {source: frame.sum() / frame.size for source, frame in diffs.items()}

            # Identify source with max diff; append to window; update last_frames
            max_source = max(diffs, key=lambda source: diffs[source], default=next(iter(new_frames)))
            window.append(max_source)

            # Update last_frames with new data
            last_frames = {source: new_frames[source] if new_frames[source] is not None
            else last_frames[source] for source in new_frames}

            # Vote proportionally based on count in window
            vote = Distribution(Counter(window))
            for key in new_frames.keys() - vote.keys():  # add missing keys
                vote[key] = 0.0
            vote.normalize()  # scale down to [0, 1]

            # Output vote distribution
            output_queue.put_nowait(vote)

        scheduler = create_periodic_event(interval=1 / 30, action=weight_sources)
        scheduler.run()
