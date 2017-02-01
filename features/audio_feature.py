from collections import deque, Counter

from util.distribution import Distribution
from util.pipeline import PipelineProcess
from util.schedule import create_periodic_event


class AudioFeature(PipelineProcess):

    def __init__(self, feature_id, audio_sources, audio_video_pair_map, window_length=10, thrash_limit=3):
        super().__init__(pipeline_id=feature_id,
                         target_function=AudioFeature.establish_process_loop,
                         params=(audio_video_pair_map, window_length, thrash_limit),
                         sources=audio_sources,
                         drop_input_frames=False,
                         drop_output_frames=True)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, audio_video_pair_map, window_length, thrash_limit):
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

            # Append max source
            source_audio = {source_id: audio_frame for source_id, audio_frame in input_queue}
            window.append(max(source_audio, key=lambda id: max(source_audio[id], default=0)))

            # Examination of sliding window
            source_count = Counter(window)
            max_choice = source_count.most_common(1)[0][0]  # returns list of pairs ala [(item, count)]

            # Consideration for thrashing
            if max_choice != last_selected:
                if time_since_switch > thrash_limit or last_selected is None:
                        last_selected = max_choice
                        time_since_switch = 0

            # Vote distribution
            selected_video = audio_video_pair_map[last_selected]
            vote = Distribution({video_source: 0 for audio_source, video_source in audio_video_pair_map})
            vote[selected_video] = 1.0
            output_queue.put(vote)

        scheduler = create_periodic_event(interval=1 / 30, action=weight_sources)
        scheduler.run()





