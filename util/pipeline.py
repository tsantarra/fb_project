from multiprocessing import Process, Queue


class PipelineFunction:
    """ This class operates as an intermediate processing point between inputs and outputs. """

    def __init__(self, target_function, params, sources, drop_frames=False):
        """ Initialize the synchronized objects and work process. """
        size = 1 if drop_frames else 0

        self.input_queue = Queue(maxsize=size)
        self.input_sources = sources

        self.output_queue = Queue(maxsize=size)
        self.output_frame = None

        self.process = Process(target=target_function,
                               args=[self.input_queue, self.output_queue] + list(params))

    def start(self):
        """ Begin the work process. """
        self.process.start()

    def update(self):
        """ Update the inputs and outputs of the function. """
        if self.input_sources:
            self.input_queue.put([source.read() for source in self.input_sources])

        if self.output_queue.empty():
            self.output_frame = None
        else:
            self.output_frame = self.output_queue.get()

    def read(self):
        """ Return the latest frame of data. """
        return self.output_frame

    def close(self):
        """ End the work process. """
        self.process.terminate()
