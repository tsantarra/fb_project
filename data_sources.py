import wave
import struct
import cv2
import sounddevice as sd
import logging


def get_camera_list():
    """ Unfortunately, it seems that OpenCV does not provide a way to acquire a
        list of active devices. There is likely a better way than iterating through
        possible device IDs and trying each.
    """
    id = 0
    cam_list = []

    while True:
        stream = cv2.VideoCapture(id)
        if stream.isOpened():
            stream.release()
            cam_list.append(id)
            id += 1
        else:
            break

    return cam_list


class DataSource:

    def update(self):
        """ Capture the next frame of data. """
        raise NotImplementedError

    def read(self):
        """ Return the latest frame of data. """
        raise NotImplementedError


class VideoStream(DataSource):

    def __init__(self, id):
        self.stream = cv2.VideoCapture(id)
        self.last_frame = None
        self.status = self.stream.isOpened()

    def update(self):
        self.status, self.last_frame = self.stream.read()

    def read(self):
        return self.last_frame

    def __del__(self):
        self.stream.release()


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


class AudioStream(DataSource):

    def __init__(self, id):
        self.id = id
        self.stream = sd.InputStream(device=id, channels=1, latency='low')
        self.stream.start()
        self.last_frame = None
        self.status = self.stream.active
        self.flag = None

    def update(self):
        self.last_frame, self.flag = self.stream.read(self.stream.read_available)
        self.status = self.stream.active

    def read(self):
        return self.last_frame

    def __del__(self):
        self.stream.stop()
        self.stream.close()


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


