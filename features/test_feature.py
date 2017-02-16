from util.distribution import Distribution
from util.pipeline import PipelineProcess, get_from_queue, get_all_from_queue
from util.schedule import create_periodic_event


class TestFeature(PipelineProcess):
    """ This class serves an example of how to design a feature using the PipelineProcess class. """

    def __init__(self, feature_id, audio_video_pairs, sources, interval=1/30):
        """ PipelineProcess handles the behind-the-scenes setup of the subprocess. Just pass the superclass
            constructor a target static method to execute as well as the relevant parameters. """
        super().__init__(pipeline_id=feature_id,
                         target_function=TestFeature.establish_process_loop,
                         params=(audio_video_pairs, interval),
                         sources=sources)

    @staticmethod
    def establish_process_loop(input_queue, output_queue, audio_video_pairs, interval):
        """ This function is passed to a separate python process with shared input and output queues. """
        # State variables to be reference by repeated process.
        video_ids = [pair[1] for pair in audio_video_pairs]

        # Define function to be called repeatedly by the scheduler.
        def weight_sources():
            """ Function to be called by looping scheduler in process. """
            # Reference to state variables, bringing them into this scope. You can also pass them
            # via action_args in create_periodic_event, but the variables must be mutable to accept
            # any changes.
            nonlocal video_ids

            # Input data for feature calculation
            input_data = get_all_from_queue(input_queue)

            # Do some calculations here.

            # Output vote distribution via the output_queue.
            vote = Distribution({vid_id: 0.0 for vid_id in video_ids})
            output_queue.put_nowait(vote)

        # Schedule repeating event and run.
        scheduler = create_periodic_event(interval=interval, action=weight_sources, action_args=())
        scheduler.run()




