from collections import deque, Counter

from util.distribution import Distribution
from util.pipeline import PipelineProcess, get_from_queue
from util.schedule import create_periodic_event


class AudioFeature(PipelineProcess):

    def __init__(self, feature_id, audio_sources, audio_video_pair_map, window_length=10):
        super().__init__(pipeline_id=feature_id,
                         target_function=AudioFeature.establish_process_loop,
                         params=(audio_video_pair_map, window_length),
                         sources=audio_sources,
                         drop_input_frames=True,
                         drop_output_frames=True)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, audio_video_pair_map, window_length):
        window = deque(maxlen=window_length)  # A sliding window containing the most active stream for each frame
        last_frames = {}                      # The last set of frames viewed
        video_ids = set(audio_video_pair_map.values())

        def weight_sources():
            # Inform Python we are using vars from the outer scope.
            nonlocal window, last_frames, video_ids

            # Initial conditions
            if not last_frames:
                input_data = get_from_queue(input_queue)
                if input_data:
                    last_frames = {source_id: frame for source_id, frame in input_data}

                return

            # Determine loudest source; append corresponding video ID to sliding window
            source_audio = {source_id: (audio_frame if audio_frame is not None else [])
                            for source_id, audio_frame in input_queue.get()}
            max_audio_id = max(source_audio, key=lambda id: max(source_audio[id], default=0))
            window.append(audio_video_pair_map[max_audio_id])

            # Vote proportionally based on count in window
            vote = Distribution(Counter(window))
            for key in video_ids - vote.keys():  # add missing keys
                vote[key] = 0.0
            vote.normalize()  # scale down to [0, 1]

            # Output vote distribution
            output_queue.put(vote)

        scheduler = create_periodic_event(interval=1 / 30, action=weight_sources)
        scheduler.run()





