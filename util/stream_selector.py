from util.distribution import Distribution


class StreamSelector:
    """ This class is responsible for aggregating the feature votes and changing output streams. """

    def __init__(self, inputs, weighted_feature_distribution, outputs, thrash_limit=30):
        self.inputs = inputs
        self.features = list(weighted_feature_distribution.keys())
        self.feature_weights = weighted_feature_distribution
        self.outputs = outputs
        self._all_input_output = set(self.inputs.audio + self.inputs.video + self.inputs.main_audio +
                                     self.features + self.outputs.audio + self.outputs.video + self.outputs.main_video)

        self.video_input_map = {stream.id: stream for stream in inputs.video}

        self.started = False

        # Considerations for thrashing (switching back and forth rapidly)
        self.last_selected = None
        self.time_since_switch = 0
        self.thrash_limit = thrash_limit

    def update(self):
        """
        Steps:
            - collect votes
            - tally
            - direct output stream
        """
        # Moved start call to first update loop.
        if not self.started:
            self.start()

        # Update all subprocesses
        for process in self._all_input_output:
            process.update()

        # Read in votes. Check votes for appropriate type.
        votes = {}
        for feature in self.features:
            votes_output = feature.read()
            if votes_output:
                votes[feature] = votes_output[-1]  # in the event of multiple votes queued, just take most recent.

        assert all(type(vote) == Distribution for vote in votes.values() if vote is not None), \
            'Vote types:' + '\n'.join([type(vote) for vote in votes])

        # Tally votes.
        tally = sum(vote * self.feature_weights[feature] for feature, vote in votes.items() if vote is not None)
        if tally == 0:  # no votes yet (common during initialization)
            return

        # Determine max vote and adjust primary output streams
        max_vote = max(tally, key=lambda v: tally[v])

        # Consideration for thrashing. Only switch if same stream selected for > thrash_limit cycles.
        self.time_since_switch += 1
        if self.last_selected is None or \
                (max_vote != self.last_selected and self.time_since_switch > self.thrash_limit):
            self.last_selected = max_vote
            self.time_since_switch = 0

            # Criteria met for switching input streams
            for video_output in self.outputs.main_video:
                video_output.set_inputs([self.video_input_map[max_vote]])

    def start(self):
        # start all sub-processes
        for process in self._all_input_output:
            process.start()

        self.started = True

    def close(self):
        # start all sub-processes
        for process in self._all_input_output:
            process.close()
