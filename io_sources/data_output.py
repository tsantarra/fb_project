
import sounddevice
import soundfile
import subprocess
import numpy
import time

from multiprocessing import Process, Queue, Value
from schedule import create_periodic_event
from pipeline_interfaces import PipelineFunction


class ReadFromOutputException(Exception):
    pass

###########################################################################################################
########################################    STREAMS     ###################################################
###########################################################################################################


class OutputVideoStream(PipelineFunction):

    def __init__(self, stream_name, input, dimensions=(640, 480)):
        self.id = str(stream_name)
        self.input = input
        self.end = Value('b', 0)

        self.queue = Queue(maxsize=1)  # allow dropped frames
        self.process = Process(target=OutputVideoStream.show_video, args=(self.id, self.queue, dimensions, self.end))

    def start(self):
        self.process.start()

    def update(self):
        if self.input.status:
            self.queue.put(self.input.read())

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    def close(self):
        self.end.value = 1

    def complete(self):
        return self.process.exitcode is not None

    @staticmethod
    def show_video(stream_id, queue, dimensions, end):
        import cv2

        def display_video_frame(queue):
            if not queue.empty():
                frame = queue.get()
                cv2.imshow(stream_id, cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)

        def check_end():
            end_val = bool(end.value)
            if end_val:
                cv2.destroyWindow(stream_id)
            return end_val

        scheduler = create_periodic_event(interval=1/30,
                                          action=display_video_frame,
                                          action_args=(queue,),
                                          halt_check=check_end)
        scheduler.run()


class OutputAudioStream(PipelineFunction):

    def __init__(self, device_id, input, sample_rate, dtype, channels=1, latency='low'):
        self.id = device_id
        self.input = input

        self.end = Value('b', 0)
        self.queue = Queue()
        self.process = Process(target=OutputAudioStream.output_audio, args=(self.id, channels, sample_rate,
                                                                            latency, dtype, self.queue, self.end))

    def start(self):
        self.process.start()

    def update(self):
        if self.input.status:
            data = self.input.read()

            if type(data) == numpy.ndarray:
                self.queue.put(data)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    def close(self):
        self.end.value = 1

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def output_audio(device_id, channels, sample_rate, latency, dtype, queue, end):
        stream = sounddevice.OutputStream(device=int(device_id), channels=channels, samplerate=sample_rate, latency=latency, dtype=dtype)
        stream.start()

        def write_audio_frames(queue):
            frame = None
            while not queue.empty():
                new_frame = queue.get()

                if frame is None:
                    frame = new_frame
                else:
                    frame = numpy.append(frame, new_frame)

            if frame is not None:
                stream.write(frame)  #.astype(dtype=dtype))

        def check_end():
            end_val = bool(end.value)
            return end_val

        scheduler = create_periodic_event(interval=1/sample_rate, action=write_audio_frames, action_args=(queue,), halt_check=check_end)
        scheduler.run()

###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################


class OutputVideoFile(PipelineFunction):

    def __init__(self, filename, input, video_fps=30, dimensions=(640, 480)):
        self.input = input

        self.last_frame = Queue(maxsize=1)
        self.end = Value('b', 0)
        self.queue = Queue()
        self.process = Process(target=OutputVideoFile.output_video, args=(filename, video_fps, dimensions, self.queue, self.last_frame, self.end))

    def start(self):
        self.process.start()

    def update(self):
        if self.input.status:
            frame = self.input.read()
            self.last_frame.put(frame)
            self.queue.put(frame)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    def close(self):
        self.process.terminate()

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def output_video(filename, video_fps, dimensions, queue, last_frame, end):

        import cv2
        stream = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), video_fps, dimensions)

        def write_video_frames():
            if queue.empty():
                if last_frame.empty():  # Only empty at the beginning of the process.
                    return
                frame = last_frame.get()
            else:
                frame = queue.get()

            stream.write(cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA))

        def check_end():
            end_val = bool(end.value)
            return end_val

        scheduler = create_periodic_event(interval=1/video_fps, action=write_video_frames, action_args=(), halt_check=check_end)
        scheduler.run()


class OutputAudioFile(PipelineFunction):

    def __init__(self, filename, input, sample_rate, channels=1):
        self.input = input

        self.end = Value('b', 0)
        self.queue = Queue(maxsize=1)
        self.process = Process(target=OutputAudioFile.output_audio, args=(filename, sample_rate, channels, self.queue, self.end))

    def start(self):
        self.process.start()

    def update(self):
        if self.input.status:
            data = self.input.read()

            if type(data) == numpy.ndarray:
                self.queue.put(data)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    def close(self):
        self.end.value = 1
        #self.process.terminate()

    def complete(self):
        return self.process.exitcode is not None

    def __del__(self):
        self.close()

    @staticmethod
    def output_audio(filename, sample_rate, channels, queue, end):
        stream = soundfile.SoundFile(filename, mode='w', samplerate=sample_rate, channels=channels)

        def write_audio_frames(queue):
            frame = None
            while not queue.empty():
                new_frame = queue.get()

                if frame is None:
                    frame = new_frame
                else:
                    frame = numpy.append(frame, new_frame)

            if frame is not None:
                stream.write(frame)  #.astype(dtype='float32'))

        def check_end():
            end_val = bool(end.value)

        scheduler = create_periodic_event(interval=1/sample_rate, action=write_audio_frames, action_args=(queue,), halt_check=check_end)
        scheduler.run()

###########################################################################################################
########################################      Join Fn     #################################################
###########################################################################################################


def join_audio_and_video(audio_filename, video_filename):
    cmd = 'ffmpeg -y -i ' + video_filename + ' -i ' + audio_filename + ' -shortest -async 1 -vsync 1 -codec copy output.avi'
    # flags  -codec copy
    subprocess.call(cmd, shell=True)