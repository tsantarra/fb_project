from util.distribution import Distribution


class StreamSelector:

    def __init__(self, inputs, feature_set, outputs):
        self.inputs = inputs
        self.features = feature_set
        self.outputs = outputs

        self.video_input_map = {stream.id: stream for stream in inputs.video}

        self.stream_vote_aggregate = None

    def update(self):
        """
        Steps:
            - collect votes
            - tally
            - direct output stream
        """
        # NOTE: DO NOT UPDATE inputs.main_audio, as it ALREADY UPDATES VIA inputs.audio
        # update output to input
        for process in self.outputs.audio + self.outputs.video + self.features + self.inputs.audio  + self.inputs.video:
            process.update()

        votes = [feature.read().data for feature in self.features]

        assert all(type(vote) == Distribution for vote in votes if vote is not None), '\n'.join([type(vote) for vote in votes])

        tally = sum(vote for vote in votes if vote is not None)
        if tally == 0:
            return  # no votes yet (common during initialization

        max_vote = max(tally, key=lambda v: tally[v])
        print(max_vote)
        for output_stream in self.outputs.video:
            output_stream.set_inputs([self.video_input_map[max_vote]])

    def start(self):
        # start all sub-processes
        for process in set(self.inputs.audio + self.inputs.video + [self.inputs.main_audio] +
                           self.features + self.outputs.audio + self.outputs.video):
            process.start()

    def close(self):
        # start all sub-processes
        for process in set(self.inputs.audio + self.inputs.video + [self.inputs.main_audio] +
                           self.features + self.outputs.audio + self.outputs.video):
            process.close()



