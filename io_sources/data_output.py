import cv2
import sounddevice
import soundfile
import subprocess
import numpy
import time

from multiprocessing import Process, Queue
from schedule import create_periodic_event


class Output:
    """ This class operates as the output interface for all audio and video feeds. """

    def write(self, data):
        """ Write the next frame of data to output destination. """
        raise NotImplementedError

    def close(self):
        """ Write the next frame of data to output destination. """
        raise NotImplementedError

###########################################################################################################
########################################    STREAMS     ###################################################
###########################################################################################################


class OutputVideoStream(Output):
    @staticmethod
    def show_video(stream_id, queue, dimensions):
        def display_video_frame(queue):
            frame = queue.get()
            if frame is not None:
                cv2.imshow(stream_id, cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)

        scheduler = create_periodic_event(interval=0.033, action=display_video_frame, action_args=(queue, ))
        scheduler.run()

    def __init__(self, stream_name, dimensions=(640, 480)):
        self.id = str(stream_name)
        self.queue = Queue(maxsize=1)
        self.process = Process(target=OutputVideoStream.show_video, args=(self.id, self.queue, dimensions))
        self.process.start()
        self.drops = 0

    def write(self, data):
        self.drops += int(not self.queue.empty())
        self.queue.put(data)

    def close(self):
        print('Dropped frames:', self.drops)
        self.process.terminate()


class OutputAudioStream(Output):
    @staticmethod
    def output_audio(device_id, channels, sample_rate, latency, dtype, queue):
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

        scheduler = create_periodic_event(interval=1/sample_rate, action=write_audio_frames, action_args=(queue,))
        scheduler.run()

    def __init__(self, device_id, sample_rate, dtype, channels=1, latency='low'):
        self.id = device_id
        self.queue = Queue()
        self.process = Process(target=OutputAudioStream.output_audio, args=(self.id, channels, sample_rate,
                                                                            latency, dtype, self.queue))
        self.process.start()

    def write(self, data):
        if type(data) == numpy.ndarray:
            self.queue.put(data)

    def close(self):
        self.process.terminate()


###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################

class OutputVideoFile(Output):
    @staticmethod
    def output_video(filename, video_fps, dimensions, queue):
        stream = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), video_fps, dimensions)

        def write_video_frames(queue):
            frame = queue.get()
            while frame is not None:
                stream.write(cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA))
                frame = queue.get()

        scheduler = create_periodic_event(interval=1/video_fps, action=write_video_frames, action_args=(queue, ))
        scheduler.run()

    def __init__(self, filename, video_fps=30, dimensions=(640, 480)):
        self.queue = Queue()
        self.process = Process(target=OutputVideoFile.output_video, args=(filename, video_fps, dimensions, self.queue))
        self.process.start()

    def write(self, data):
        self.queue.put(data)

    def close(self):
        self.process.terminate()


class OutputAudioFile(Output):
    @staticmethod
    def output_audio(filename, sample_rate, channels, queue):
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

        scheduler = create_periodic_event(interval=1/sample_rate, action=write_audio_frames, action_args=(queue,))
        scheduler.run()

    def __init__(self, filename, sample_rate, channels=1):
        self.queue = Queue(maxsize=1)
        self.process = Process(target=OutputAudioFile.output_audio, args=(filename, sample_rate, channels, self.queue))
        self.process.start()

    def write(self, data):
        if type(data) == numpy.ndarray:
            self.queue.put(data)

    def close(self):
        self.process.terminate()

###########################################################################################################
########################################      Join Fn     #################################################
###########################################################################################################


def join_audio_and_video(audio_filename, video_filename):
    cmd = 'ffmpeg -y -i ' + video_filename + ' -i ' + audio_filename + ' -shortest -async 1 -vsync 1 -codec copy output.avi'
    # flags  -codec copy
    subprocess.call(cmd, shell=True)