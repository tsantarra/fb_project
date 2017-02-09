from itertools import zip_longest
from multiprocessing import Process, Queue
from collections import namedtuple
from queue import Empty

PipelineData = namedtuple('PipelineData', ['source_id', 'data'])


def get_from_queue(queue):
    try:
        return queue.get_nowait()
    except Empty:
        return None


def get_all_from_queue(queue):
    data = []
    while True:
        try:
            data.append(queue.get_nowait())
        except Empty:
            return data


class PipelineProcess:
    """ This class operates as an intermediate processing point between inputs and outputs. """

    def __init__(self, pipeline_id, target_function, params, sources, drop_input_frames=False, drop_output_frames=False):
        """ Initialize the synchronized objects and work process. """
        self.id = pipeline_id

        self._input_sources = sources
        self._input_queue = Queue(maxsize=int(drop_input_frames))

        self._output = PipelineData(self.id, None)
        self._output_queue = Queue(maxsize=int(drop_output_frames))

        self._process = Process(target=target_function,
                                args=[self._input_queue, self._output_queue] + list(params))

    def set_inputs(self, sources):
        """ Overwrites the input sources. Used for changing pipeline structure live. """
        self._input_sources = sources

    def start(self):
        """ Begin the work process. """
        self._process.start()

    def update(self):
        """ Update the inputs and outputs of the function. """
        if self._input_sources:
            # simultaneous input from all sources
            self._input_queue.put([source.read() for source in self._input_sources])

        self._output = PipelineData(self.id, get_from_queue(self._output_queue))

    def read(self):
        """ Return the latest frame of data. """
        return self._output

    def close(self):
        """ End the work process. """
        self._process.terminate()
