from collections import deque, Counter

import cv2

from util.distribution import Distribution
from util.pipeline import PipelineProcess, get_from_queue
from util.schedule import create_periodic_event


class VideoMovementFeature(PipelineProcess):
    """
    Votes for a video stream based on which stream has the most pairwise frame differences within a sliding window.
    """

    def __init__(self, feature_id, video_sources, window_length=10):
        super().__init__(pipeline_id=feature_id,
                         target_function=VideoMovementFeature.establish_process_loop,
                         params=(window_length,),
                         sources=video_sources,
                         drop_input_frames=True,
                         drop_output_frames=True)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, window_length):
        window = deque(maxlen=window_length)  # A sliding window containing the most active stream for each frame
        last_frames = {}                      # The last set of frames viewed

        def weight_sources():
            # Inform Python we are using vars from the outer scope.
            nonlocal window, last_frames

            # Initial conditions
            if not last_frames:
                input_data = get_from_queue(input_queue)
                if input_data:
                    last_frames = {source_id: frame for source_id, frame in input_data}

                return

            # Collect new frames
            new_frames = {source_id: frame for source_id, frame in input_queue.get()}

            # Calculated diffs between new and last frames
            diffs = {source: cv2.absdiff(new_frames[source], last_frames[source]) for source in new_frames
                     if (new_frames[source] is not None and last_frames[source] is not None)}
            diffs = {source: cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] for source, diff in diffs.items()}
            diffs = {source: frame.sum()/frame.size for source, frame in diffs.items()}

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
            output_queue.put(vote)

        scheduler = create_periodic_event(interval=1 / 30, action=weight_sources)
        scheduler.run()
