from collections import deque, Counter
import itertools

from util.distribution import Distribution
from util.pipeline import PipelineProcess, get_all_from_queue
from util.schedule import create_periodic_event


class AudioFeature(PipelineProcess):

    def __init__(self, feature_id, audio_sources, audio_video_pair_map, window_length=10):
        super().__init__(pipeline_id=feature_id,
                         target_function=AudioFeature.establish_process_loop,
                         params=(audio_video_pair_map, window_length),
                         sources=audio_sources)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, audio_video_pair_map, window_length):
        window = deque(maxlen=window_length)  # A sliding window containing the most active stream for each frame
        video_ids = set(audio_video_pair_map.values())

        def weight_sources():
            # Inform Python we are using vars from the outer scope.
            nonlocal window, video_ids, audio_video_pair_map

            source_audio = {source_id: [] for source_id in audio_video_pair_map}
            for update_step in get_all_from_queue(input_queue):
                for source_id, audio_frame_list in update_step.items():
                    source_audio[source_id] += list(itertools.chain.from_iterable(audio_frame_list))

            # Determine loudest source; append corresponding video ID to sliding window
            max_audio_id = max(source_audio, key=lambda id: max(source_audio[id], default=0))
            window.append(audio_video_pair_map[max_audio_id])

            # Vote proportionally based on count in window
            vote = Distribution(Counter(window))
            for key in video_ids - vote.keys():  # add missing keys
                vote[key] = 0.0
            vote.normalize()  # scale down to [0, 1]

            # Output vote distribution
            output_queue.put_nowait(vote)

        scheduler = create_periodic_event(interval=1 / 30, action=weight_sources)
        scheduler.run()





