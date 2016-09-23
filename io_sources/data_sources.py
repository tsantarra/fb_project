import wave
import struct
import cv2
import sounddevice
import logging
import numpy

from multiprocessing import Process, Queue
from schedule import create_periodic_event


class DataSource:
    """ This class operates as the interface for all audio and video sources. """

    def update(self):
        """ Capture the next frame of data. """
        raise NotImplementedError

    def read(self):
        """ Return the latest frame of data. """
        raise NotImplementedError

    def start(self):
        raise NotImplementedError


###########################################################################################################
########################################    STREAMS     ###################################################
###########################################################################################################

class AudioStream(DataSource):
    @staticmethod
    def stream_audio(device_id, sample_rate, dtype, queue):
        """
            This function is given to a sub-process for execution. It functions by opening an InputStream, then using
            a scheduler to periodically grab frames, placing them in the synced Queue from the AudioStream instance.
        """
        stream = sounddevice.InputStream(device=device_id, channels=1, samplerate=sample_rate, latency='low', dtype=dtype)
        stream.start()

        def grab_audio_frames(queue, stream):
            frame, flag = stream.read(stream.read_available)
            queue.put((frame, flag, stream.active))

        scheduler = create_periodic_event(interval=0.001, action=grab_audio_frames, action_args=(queue, stream))
        scheduler.run()

    def __init__(self, device_id, sample_rate, dtype):
        self.id = device_id
        self.last_frame = []
        self.status = True
        self.flag = None

        self.queue = Queue()
        self.process = Process(target=AudioStream.stream_audio, args=(device_id, sample_rate, dtype, self.queue))
        self.process.start()

    def update(self):
        frame = None
        while not self.queue.empty():
            new_frame, self.flag, self.status = self.queue.get()

            if frame is None:
                frame = new_frame
            else:
                frame = numpy.append(frame, new_frame)

        if frame is not None:
            self.last_frame = frame

    def read(self):
        return self.last_frame

    def __del__(self):
        self.process.terminate()


class VideoStream(DataSource):
    @staticmethod
    def stream_video(device_id, queue, interval=0.015):
        """
            This function is given to a sub-process for execution. It functions by opening an InputStream, then using
            a scheduler to periodically grab frames, placing them in the synced Queue from the VideoStream instance.
        """
        stream = cv2.VideoCapture(device_id)

        def grab_video_frame(queue, stream):
            status, frame = stream.read()
            queue.put((status, frame))

        scheduler = create_periodic_event(interval=interval, action=grab_video_frame, action_args=(queue, stream))
        scheduler.run()

    def __init__(self, device_id, input_interval=0.015):
        self.id = device_id
        self.interval = input_interval
        self.status = False
        self.last_frame = None

        self.queue = Queue(maxsize=1)
        self.process = Process(target=VideoStream.stream_video, args=(device_id, self.queue, input_interval))
        self.process.start()

    def update(self):
        if not self.queue.empty():
            self.status, self.last_frame = self.queue.get()

    def read(self):
        return self.last_frame

    def __del__(self):
        self.process.terminate()


###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################


class AudioFile(DataSource):
    @staticmethod
    def read_from_file(filename, queue):
        file_stream = wave.open(filename, 'rb')
        file_length = file_stream.getnframes()
        frames_per_second = file_stream.getframerate()

        chunk_size = 1000
        interval = 1.0 / frames_per_second * chunk_size

        def read_frames(stream, queue):
            raw_data = stream.readframes(nframes=chunk_size)

            # for compatibility with sounddevice output, need numpy array
            # source: http://stackoverflow.com/questions/30550212/raw-numpy-array-from-real-time-network-audio-stream-in-python
            numpy_array = numpy.fromstring(raw_data, '<h')
            queue.put(numpy_array) #.astype('float32', copy=True))

            # end criteria? check with file length? Keep track of position somehow?

        scheduler = create_periodic_event(interval=1/frames_per_second, action=read_frames,
                                          action_args=(file_stream, queue,))
        scheduler.run()

    def __init__(self, filename):
        self.last_frame = None
        self.status = False

        self.queue = Queue()
        self.process = Process(target=AudioFile.read_from_file, args=(filename, self.queue))
        self.process.start()

    def update(self):
        frame = None
        self.status = False

        while not self.queue.empty():
            new_frame = self.queue.get()

            if frame is None:
                frame = new_frame
            else:
                frame = numpy.append(frame, new_frame)

        if frame is not None:
            self.last_frame = frame
            self.status = True
        else:
            self.last_frame = None

    def read(self):
        return self.last_frame

    def __del__(self):
            self.process.terminate()


class VideoFile(DataSource):
    @staticmethod
    def read_file(filename, queue):
        stream = cv2.VideoCapture(filename)
        frame_rate = stream.get(cv2.CAP_PROP_FPS)

        def read_frame(queue):
            status, frame = stream.read()
            if status:
                queue.put(frame)

        scheduler = create_periodic_event(interval=1/frame_rate, action=read_frame, action_args=(queue,))
        scheduler.run()

    def __init__(self, filename):
        self.last_frame = None
        self.status = False

        self.queue = Queue()
        self.process = Process(target=VideoFile.read_file, args=(filename, self.queue))
        self.process.start()

    def update(self):
        if not self.queue.empty():
            self.last_frame = self.queue.get()
            self.status = True
        else:
            self.status = False

    def read(self):
        return self.last_frame

    def __del__(self):
        self.process.terminate()