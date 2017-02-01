import subprocess

import numpy
import sounddevice
import soundfile

from util.pipeline import PipelineProcess
from util.schedule import create_periodic_event


class ReadFromOutputException(Exception):
    pass

###########################################################################################################
########################################    STREAMS     ###################################################
###########################################################################################################


class OutputVideoStream(PipelineProcess):

    def __init__(self, stream_id, input_stream, dimensions=(640, 480), interval=1 / 30):
        super().__init__(pipeline_id='OVS-' + str(stream_id),
                         target_function=OutputVideoStream.show_video,
                         params=(self.id, dimensions, interval),
                         sources=[input_stream],
                         drop_frames=True)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output_files pipeline function.' + str(self.__class__))

    @staticmethod
    def show_video(input_queue, output_queue, stream_id, dimensions, interval):
        import cv2

        def display_video_frame():
            if not input_queue.empty():
                inputs = input_queue.get()
                frame = inputs[0]

                if type(frame) == numpy.ndarray:
                    cv2.imshow(stream_id, cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA))
                    cv2.waitKey(1)

        scheduler = create_periodic_event(interval=interval, action=display_video_frame)
        scheduler.run()


class OutputAudioStream(PipelineProcess):

    def __init__(self, device_id, input_stream, sample_rate, dtype, channels=1, latency='low', interval=1/30):
        self.input_stream = input_stream

        super().__init__(pipeline_id='OAS-' + str(device_id),
                         target_function=OutputAudioStream.output_audio,
                         params=(device_id, channels, sample_rate, latency, dtype, interval),
                         sources=[input_stream],
                         drop_frames=True)

    def update(self):
        data = self.input_stream.read()

        if type(data) == numpy.ndarray:
            self._input_queue.put(data)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output_files pipeline function.' + str(self.__class__))

    @staticmethod
    def output_audio(input_queue, output_queue, device_id, channels, sample_rate, latency, dtype, interval):
        stream = sounddevice.OutputStream(device=int(device_id), channels=channels, samplerate=sample_rate, latency=latency, dtype=dtype)
        stream.start()

        def write_audio_frames():
            frame = None
            while not input_queue.empty():
                new_frame = input_queue.get()

                if frame is None:
                    frame = new_frame
                else:
                    frame = numpy.append(frame, new_frame)

            if frame is not None:
                stream.write(frame)

        scheduler = create_periodic_event(interval=interval, action=write_audio_frames)
        scheduler.run()

###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################


class OutputVideoFile(PipelineProcess):

    def __init__(self, filename, input_stream, video_fps=30, dimensions=(640, 480), interval=1/30):
        self.filename = filename
        self.input_stream = input_stream

        super().__init__(pipeline_id='OVF-' + str(filename),
                         target_function=OutputVideoFile.output_video,
                         params=(filename, video_fps, dimensions, interval),
                         sources=None,
                         drop_frames=False)

    def update(self):
        frame = self.input_stream.read()

        if type(frame) == numpy.ndarray:
            self._input_queue.put(frame)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output_files pipeline function.' + str(self.__class__))

    @staticmethod
    def output_video(input_queue, output_queue, filename, video_fps, dimensions, interval):
        import cv2
        stream = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), video_fps, dimensions)

        # Necessary part: we need a container to store the last frame, which we can duplicate when we don't have
        # new frames incoming. This avoids dropping frames in a video where we can't adjust the fps.
        last_frame = [numpy.zeros((480, 640, 3), dtype='uint8')]

        def write_video_frames():
            if not input_queue.empty():
                frame = input_queue.get()
                last_frame[0] = frame
            else:
                frame = last_frame[0]

            stream.write(cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA))

        scheduler = create_periodic_event(interval=interval, action=write_video_frames)
        scheduler.run()


class OutputAudioFile(PipelineProcess):

    def __init__(self, filename, input_stream, sample_rate, channels=1, interval=1/30):
        self.input_stream = input_stream

        super().__init__(pipeline_id='OAF-' + str(filename),
                         target_function=OutputAudioFile.output_audio,
                         params=(filename, sample_rate, channels, interval),
                         sources=[input_stream],
                         drop_frames=True)

    def update(self):
        data = self.input_stream.read()

        if type(data) == numpy.ndarray:
            self._input_queue.put(data)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output_files pipeline function.' + str(self.__class__))

    @staticmethod
    def output_audio(input_queue, output_queue, filename, sample_rate, channels, interval):
        stream = soundfile.SoundFile(filename, mode='w', samplerate=sample_rate, channels=channels)

        def write_audio_frames():
            frame = None
            while not input_queue.empty():
                new_frame = input_queue.get()

                if frame is None:
                    frame = new_frame
                else:
                    frame = numpy.append(frame, new_frame)

            if frame is not None:
                stream.write(frame)

        scheduler = create_periodic_event(interval=interval, action=write_audio_frames)
        scheduler.run()

###########################################################################################################
########################################      Join Fn     #################################################
###########################################################################################################


def join_audio_and_video(audio_filename, video_filename):
    cmd = 'ffmpeg -y -i ' + video_filename + ' -i ' + audio_filename + ' -shortest -async 1 -vsync 1 -codec copy output_files/output_files.avi'
    # flags  -codec copy
    subprocess.call(cmd, shell=True)