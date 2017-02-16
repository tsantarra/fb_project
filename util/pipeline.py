from multiprocessing import Process, Manager
from collections import namedtuple
from queue import Empty

PipelineOutput = namedtuple('PipelineOutput', ['source_id', 'data'])


def get_all_from_queue(queue):
    """ Returns all items from the queue. """
    data = []
    while True:
        try:
            data.append(queue.get_nowait())
        except Empty:
            return data


class PipelineProcess:
    """ This class operates as an intermediate processing point between inputs and outputs. """

    def __init__(self, pipeline_id, target_function, params, sources):
        """ Initialize the synchronized objects and work process. """
        self.id = pipeline_id
        self._process_manager = Manager()

        self._input_sources = {source.id: source for source in sources}
        self._input_queue = self._process_manager.Queue(maxsize=0)

        self._output = []
        self._output_queue = self._process_manager.Queue(maxsize=0)

        self._process = Process(target=target_function,
                                args=[self._input_queue, self._output_queue] + list(params))

    def set_inputs(self, sources):
        """ Overwrites the input sources. Used for changing pipeline structure live. """
        self._input_sources = {source.id: source for source in sources}

    def start(self):
        """ Begin the work process. """
        self._process.start()

    def update(self):
        """ Update the inputs and outputs of the function. """
        if self._input_sources:
            input_data = {source_id: source.read() for source_id, source in self._input_sources.items()}
            if input_data:
                self._input_queue.put_nowait(input_data)

        self._output = get_all_from_queue(self._output_queue)

    def read(self):
        """ Return the latest frame of data. """
        return self._output

    def close(self):
        """ End the work process. """
        self._process.terminate()
