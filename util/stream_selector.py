from util.distribution import Distribution


class StreamSelector:

    def __init__(self, inputs, feature_set, outputs):
        self.inputs = inputs
        self.features = feature_set
        self.outputs = outputs

        self.stream_vote_aggregate = None

    def update(self):
        """
        Steps:
            - collect votes
            - tally
            - direct output stream
        """
        votes = [feature.read() for feature in self.features]
        assert all(type(vote) == Distribution for vote in votes)

        tally = sum(votes)
        max_vote = max(tally, key=lambda v: tally[v])

        # TODO

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



