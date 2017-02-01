from collections import deque, Counter

import cv2

from util.distribution import Distribution
from util.pipeline import PipelineProcess
from util.schedule import create_periodic_event


class VideoMovementFeature(PipelineProcess):
    """
    Votes for a video stream based on which stream has the most pairwise frame differences within a sliding window.
    """

    def __init__(self, feature_id, video_sources, window_length=10, thrash_limit=3):
        super().__init__(pipeline_id=feature_id,
                         target_function=VideoMovementFeature.establish_process_loop,
                         params=(window_length, thrash_limit),
                         sources=video_sources,
                         drop_input_frames=False,
                         drop_output_frames=True)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, window_length, thrash_limit):
        window = deque(maxlen=window_length)  # A sliding window containing the most active stream for each frame
        time_since_switch = 0                 # A count of how many frames it has been since the vote switched
        last_selected = None                  # The last selected stream
        last_frames = {}                      # The last set of frames viewed

        def weight_sources():
            # Inform Python we are using vars from the outer scope.
            nonlocal window, time_since_switch, last_selected, last_frames

            # Initial conditions
            if not last_frames:
                last_frames = {source_id: frame for source_id, frame in input_queue}
                return
            else:
                # Progress tracking vars
                time_since_switch += 1

            # Collect new frames
            new_frames = {source_id: frame for source_id, frame in input_queue}

            # Calculated diffs between new and last frames
            diffs = {source: cv2.absdiff(new_frames[source], last_frames[source]) for source in new_frames
                     if (new_frames[source] is not None and last_frames[source] is not None)}
            diffs = {source: cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] for source, diff in diffs.items()}
            diffs = {source: frame.sum() for source, frame in diffs.items()}

            # Identify source with max diff; append to window; update last_frames
            max_source = max(diffs, key=lambda source: diffs[source], default=next(iter(new_frames)))
            window.append(max_source)
            last_frames = new_frames

            # Examination of sliding window
            source_count = Counter(window)
            max_choice = source_count.most_common(n=1)[0][0]  # returns list of pairs ala [(item, count)]

            # Consideration for thrashing
            if max_choice != last_selected:
                if time_since_switch > thrash_limit or last_selected is None:
                    last_selected = max_choice
                    time_since_switch = 0

            # Output vote distribution
            output_queue.put(Distribution({source_id: int(source_id == last_selected) for source_id in last_frames}))

        scheduler = create_periodic_event(interval=1 / 30, action=weight_sources)
        scheduler.run()
