import wave
import struct
import cv2
import sounddevice
import logging
import numpy

from multiprocessing import Process, Queue
from schedule import create_periodic_event


class DataSource:

    def update(self):
        """ Capture the next frame of data. """
        raise NotImplementedError

    def read(self):
        """ Return the latest frame of data. """
        raise NotImplementedError


###########################################################################################################
########################################      AUDIO     ###################################################
###########################################################################################################

def stream_audio(device_id, queue, interval=0.001):
    stream = sounddevice.InputStream(device=device_id, channels=1, latency='low')
    stream.start()

    def grab_audio_frames(queue, stream):
        frame, flag = stream.read(stream.read_available)
        queue.put((frame, flag, stream.active))

    scheduler = create_periodic_event(interval=0.001, action=grab_audio_frames, action_args=(queue, stream))
    scheduler.run()


class AudioStream:

    def __init__(self, device_id, input_interval=0.001):
        self.id = device_id
        self.interval = input_interval
        self.last_frame = []
        self.status = True
        self.flag = None

        self.queue = Queue()
        self.process = Process(target=stream_audio, args=(device_id, self.queue, input_interval))
        self.process.start()

    def update(self):
        frame = None
        while not self.queue.empty():
            new_frame, self.flag, self.status = self.queue.get()

            if frame is None:
                frame = new_frame
            else:
                frame = numpy.append(frame, new_frame)

            self.last_frame = frame

    def read(self):
        return self.last_frame

    def __del__(self):
        self.process.terminate()


###########################################################################################################
########################################      VIDEO     ###################################################
###########################################################################################################


def stream_video(device_id, queue, interval=0.015):
    stream = cv2.VideoCapture(device_id)

    def grab_video_frame(queue, stream):
        status, frame = stream.read()
        queue.put((status, frame))

    scheduler = create_periodic_event(interval=interval, action=grab_video_frame, action_args=(queue, stream))
    scheduler.run()


class VideoStream:

    def __init__(self, device_id, input_interval=0.015):
        self.id = device_id
        self.interval = input_interval
        self.status = False
        self.last_frame = None

        self.queue = Queue(maxsize=1)
        self.process = Process(target=stream_video, args=(device_id, self.queue, input_interval))
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

    def __init__(self, filename, frames_per_tick=1):
        self.stream = wave.open(filename, 'r')
        self.last_frame = None
        self.status = None
        self.file_length = self.stream.getnframes()
        self.position = 0
        self.frames_per_tick = frames_per_tick

    def update(self):
        if self.position > self.file_length:
            self.last_frame = None
            return

        raw_data = self.stream.readframes(self.frames_per_tick)
        self.last_frame = int(struct.unpack("<h", raw_data)[0])
        self.position += self.frames_per_tick

    def read(self):
        return self.last_frame

    def __del__(self):
        self.stream.close()


class VideoFile(DataSource):

    def __init__(self, filename):
        self.stream = cv2.VideoCapture(filename)
        self.last_frame = None
        self.status = self.stream.isOpened()

    def update(self):
        self.status, self.last_frame = self.stream.read()

    def read(self):
        return self.last_frame

    def __del__(self):
        self.stream.release()




