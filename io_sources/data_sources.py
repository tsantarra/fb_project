import wave
import numpy
import sounddevice

from collections import Mapping

from util.pipeline import PipelineProcess, PipelineData
from util.schedule import create_periodic_event

from queue import Empty

###########################################################################################################
########################################    STREAMS     ###################################################
###########################################################################################################


class InputAudioStream(PipelineProcess):

    def __init__(self, device_id, sample_rate, dtype, interval=1/30):
        self.source_id = device_id
        super().__init__(pipeline_id='AS-' + str(device_id),
                         target_function=InputAudioStream.stream_audio,
                         params=(device_id, sample_rate, dtype, interval),
                         sources=None,
                         drop_input_frames=True,
                         drop_output_frames=True)

    def update(self):
        """ Do not update input queue, only output queue. """
        # Note: Could possibly drop frames here if needed.
        frames = []
        while True:
            try:
                frames.append(self._output_queue.get_nowait())
            except Empty:
                break

        frame = None
        for audio_frame in frames:
            if frame is None:
                frame = audio_frame
            else:
                frame = numpy.append(frame, audio_frame)

        self._output = PipelineData(self.id, frame)

    @staticmethod
    def stream_audio(input_queue, output_queue, device_id, sample_rate, dtype, interval):
        """
            This function is given to a sub-process for execution. It functions by opening an InputStream, then using
            a scheduler to periodically grab frames, placing them in the synced Queue from the AudioStream instance.
        """
        stream = sounddevice.InputStream(device=device_id, channels=1, samplerate=sample_rate, latency='low', dtype=dtype)
        stream.start()

        def grab_audio_frames():
            frame, flag = stream.read(stream.read_available)
            output_queue.put(frame)

        scheduler = create_periodic_event(interval=1/30, action=grab_audio_frames)
        scheduler.run()


class InputVideoStream(PipelineProcess):

    def __init__(self, device_id, input_interval=1/30):
        self.source_id = device_id
        super().__init__(pipeline_id='VS-' + str(device_id),
                         target_function=InputVideoStream.stream_video,
                         params=(device_id, input_interval),
                         sources=None,
                         drop_input_frames=True,
                         drop_output_frames=True)

    @staticmethod
    def stream_video(input_queue, output_queue, device_id, interval):
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
                    output_queue.put(frame)

        scheduler = create_periodic_event(interval=interval, action=grab_video_frame)
        scheduler.run()


###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################


class InputAudioFile(PipelineProcess):

    def __init__(self, filename, interval=1/30):
        self.source_id = filename
        super().__init__(pipeline_id='AF-' + filename,
                         target_function=InputAudioFile.read_from_file,
                         params=(filename, interval),
                         sources=None,
                         drop_input_frames=False,
                         drop_output_frames=False)

    def update(self):
        # Note: Could possibly drop frames here if needed.
        frame = None
        while not self._output_queue.empty():
            new_frame, flag, status = self._output_queue.get()

            if frame is None:
                frame = new_frame
            else:
                frame = numpy.append(frame, new_frame)

        self.output_frame = frame

    @staticmethod
    def read_from_file(input_queue, output_queue, filename, interval):
        stream = wave.open(filename, 'rb')

        frames_per_second = stream.getframerate()
        chunk_size = interval*frames_per_second

        def read_frames():
            # for compatibility with sound device output, need numpy array
            # source: http://stackoverflow.com/questions/30550212/
            # raw-numpy-array-from-real-time-network-audio-stream-in-python
            raw_data = stream.readframes(nframes=chunk_size)
            numpy_array = numpy.fromstring(raw_data, '<h')
            output_queue.put(numpy_array)

        scheduler = create_periodic_event(interval=interval, action=read_frames)
        scheduler.run()


class InputVideoFile(PipelineProcess):

    def __init__(self, filename, interval=1/30):
        self.source_id = filename
        super().__init__(pipeline_id='VF-' + filename,
                         target_function=InputVideoFile.read_file(),
                         params=(filename, interval),
                         sources=None,
                         drop_input_frames=False,
                         drop_output_frames=False)

    @staticmethod
    def read_file(input_queue, output_queue, filename, interval):
        import cv2
        stream = cv2.VideoCapture(filename)
        # frame_rate = stream.get(cv2.CAP_PROP_FPS)

        def read_frame():
            status, frame = stream.read()
            if status:
                output_queue.put(status, frame)

        scheduler = create_periodic_event(interval=interval, action=read_frame)
        scheduler.run()
