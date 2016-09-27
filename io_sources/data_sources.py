import wave
import struct
import sounddevice
import logging
import numpy
import time

from multiprocessing import Process, Queue, Value
from schedule import create_periodic_event
from pipeline_interfaces import PipelineFunction

###########################################################################################################
########################################    STREAMS     ###################################################
###########################################################################################################


class InputAudioStream(PipelineFunction):

    def __init__(self, device_id, sample_rate, dtype):
        self.id = device_id
        self.last_frame = []
        self.status = True
        self.flag = None

        self.end = Value('b', 1)
        self.queue = Queue()
        self.process = Process(target=InputAudioStream.stream_audio, args=(device_id, sample_rate, dtype, self.queue, self.end))

    def start(self):
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
            self.status = True
            self.last_frame = frame
        else:
            self.status = False
            self.last_frame = None

    def read(self):
        return self.last_frame

    def close(self):
        self.end.value = 1

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def stream_audio(device_id, sample_rate, dtype, queue, end):
        """
            This function is given to a sub-process for execution. It functions by opening an InputStream, then using
            a scheduler to periodically grab frames, placing them in the synced Queue from the AudioStream instance.
        """
        stream = sounddevice.InputStream(device=device_id, channels=1, samplerate=sample_rate, latency='low', dtype=dtype)
        stream.start()

        def grab_audio_frames(queue, stream):
            frame, flag = stream.read(stream.read_available)
            queue.put((frame, flag, stream.active))

        def check_end():
            end_val = bool(end.value)
            return end_val

        scheduler = create_periodic_event(interval=0.001, action=grab_audio_frames, action_args=(queue, stream), halt_check=check_end)
        scheduler.run()


class InputVideoStream:

    def __init__(self, device_id, input_interval=1/60):
        self.id = device_id
        self.interval = input_interval
        self.status = False
        self.last_frame = None

        self.start_time = time.time()
        self.frames = 0

        self.end = Value('b', 0)
        self.queue = Queue()
        self.process = Process(target=InputVideoStream.stream_video, args=(device_id, self.queue, self.end, input_interval))

    def start(self):
        self.process.start()

    def update(self):
        if not self.queue.empty():
            self.status, frame = self.queue.get()

            self.frames += 1
            print(self.id, time.time() - self.start_time, self.frames)

            if self.status:
                self.last_frame = frame
        else:
            self.status = False
            self.last_frame = None

    def read(self):
        return self.last_frame

    def close(self):
        self.end.value = 1
        self.process.terminate()

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def stream_video(device_id, queue, end, interval):
        """
            This function is given to a sub-process for execution. It functions by opening an InputStream, then using
            a scheduler to periodically grab frames, placing them in the synced Queue from the VideoStream instance.
        """
        import cv2
        stream = cv2.VideoCapture(device_id)

        def grab_video_frame():
            if stream.isOpened():
                status, frame = stream.read()
                if status:
                    queue.put((status, frame))

        def check_end():
            end_val = bool(end.value)
            if end_val:
                queue.clear()
                stream.release()
            return end_val

        scheduler = create_periodic_event(interval=interval, action=grab_video_frame, action_args=(), halt_check=check_end)
        scheduler.run(blocking=True)


###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################


class InputAudioFile(PipelineFunction):

    def __init__(self, filename):
        self.last_frame = None
        self.status = False

        self.end = Value('b', 0)
        self.queue = Queue()
        self.process = Process(target=InputAudioFile.read_from_file, args=(filename, self.queue, self.end))

    def start(self):
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
            self.status = True
            self.last_frame = frame
        else:
            self.status = False
            self.last_frame = None

    def read(self):
        return self.last_frame

    def close(self):
        self.end.value = 1
        #self.process.terminate()

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def read_from_file(filename, queue, end):
        file_stream = wave.open(filename, 'rb')
        file_length = file_stream.getnframes()
        frames_per_second = file_stream.getframerate()

        chunk_size = 1000
        interval = 1.0 / frames_per_second * chunk_size

        def read_frames(stream, queue):
            raw_data = stream.readframes(nframes=chunk_size)

            # for compatibility with sound device output, need numpy array
            # source: http://stackoverflow.com/questions/30550212/raw-numpy-array-from-real-time-network-audio-stream-in-python
            numpy_array = numpy.fromstring(raw_data, '<h')
            queue.put(numpy_array)  # .astype('float32', copy=True))

        def check_end():
            end_val = bool(end.value)
            return end_val

        scheduler = create_periodic_event(interval=1/frames_per_second, action=read_frames,
                                          action_args=(file_stream, queue,), halt_check=check_end)
        scheduler.run()


class InputVideoFile(PipelineFunction):

    def __init__(self, filename):
        self.last_frame = None
        self.status = False

        self.end = Value('b', 0)
        self.queue = Queue()
        self.process = Process(target=InputVideoFile.read_file, args=(filename, self.queue, self.end))

    def start(self):
        self.process.start()

    def update(self):
        if not self.queue.empty():
            self.status, self.last_frame = self.queue.get()
        else:
            self.status = False
            self.last_frame = None

    def read(self):
        return self.last_frame

    def close(self):
        self.end.value = 1
        #self.process.terminate()

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def read_file(filename, queue, end):
        import cv2
        stream = cv2.VideoCapture(filename)
        frame_rate = stream.get(cv2.CAP_PROP_FPS)

        def read_frame(queue):
            status, frame = stream.read()
            if status:
                queue.put(status, frame)

        def check_end():
            end_val = bool(end.value)
            return end_val

        scheduler = create_periodic_event(interval=1 / frame_rate, action=read_frame, action_args=(queue,), halt_check=check_end)
        scheduler.run()
