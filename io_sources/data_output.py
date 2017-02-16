import numpy
import sounddevice
import soundfile

from util.pipeline import PipelineProcess, get_all_from_queue
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
                         params=(stream_id, dimensions, interval),
                         sources=[input_stream])

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    @staticmethod
    def show_video(input_queue, output_queue, stream_id, dimensions, interval):
        import cv2
        last_frame = numpy.zeros((dimensions[1], dimensions[0], 3), dtype='uint8')

        def display_video_frame():
            nonlocal last_frame

            # Grab waiting inputs. If none, return.
            queue_input = get_all_from_queue(input_queue)
            if not queue_input:
                return

            # Ignore all but the last input, as we need to keep up with live input.
            last_update = queue_input[-1]
            frame_list = next(iter(last_update.values()))

            # Update and display the last frame.
            if frame_list:
                last_frame = frame_list[-1]
                cv2.imshow(stream_id, cv2.resize(last_frame, dimensions, interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)

        scheduler = create_periodic_event(interval=interval, action=display_video_frame)
        scheduler.run()


class OutputTiledVideoStream(PipelineProcess):

    def __init__(self, stream_id, inputs, dimensions=(640, 480), interval=1 / 30):
        super().__init__(pipeline_id='OVS-' + str(stream_id),
                         target_function=OutputTiledVideoStream.show_video,
                         params=(stream_id, [input.id for input in inputs], dimensions, interval),
                         sources=inputs)

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    @staticmethod
    def show_video(input_queue, output_queue, stream_id, input_ids, dimensions, interval):
        import cv2, math, numpy

        # Calculate dimensions for output grid frames
        scale_factor = math.ceil(math.sqrt(len(input_ids)))
        width, height = int(dimensions[0]/scale_factor), int(dimensions[1]/scale_factor)

        # Create empty frames for open spots in grid
        padding_frames = [numpy.zeros((height, width, 3), dtype='uint8')
                          for _ in range(scale_factor**2 - len(input_ids))]

        last_frames = {input_id: numpy.zeros((height, width, 3), dtype='uint8')
                       for input_id in input_ids}

        def display_video_frame():
            nonlocal last_frames, scale_factor, height, width, padding_frames

            # grab new frames from input
            new_frames = {}
            for update_step in get_all_from_queue(input_queue):
                for source_id, frame_list in update_step.items():
                    if frame_list and type(frame_list[-1]) == numpy.ndarray:
                        new_frames[source_id] = frame_list[-1]

            last_frames.update(new_frames)

            # Pad in extra frames for complete square
            frames = list(frame if frame.shape == (height, width, 3)
                          else cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                          for frame in last_frames.values())
            frames += padding_frames

            # Merge frames into one large frame
            rows = []
            for row in range(scale_factor):
                rows.append(numpy.hstack(tuple(frames[row*scale_factor + i] for i in range(scale_factor))))
            combined = numpy.vstack(tuple(rows))

            # Display
            cv2.imshow(stream_id, combined)
            cv2.waitKey(1)

        scheduler = create_periodic_event(interval=interval, action=display_video_frame)
        scheduler.run()


class OutputAudioStream(PipelineProcess):

    def __init__(self, device_id, input_stream, sample_rate, dtype, channels=1, latency='low', interval=1/30):
        super().__init__(pipeline_id='OAS-' + str(device_id),
                         target_function=OutputAudioStream.output_audio,
                         params=(device_id, channels, sample_rate, latency, dtype, interval),
                         sources=[input_stream])

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    @staticmethod
    def output_audio(input_queue, output_queue, device_id, channels, sample_rate, latency, dtype, interval):
        stream = sounddevice.OutputStream(device=int(device_id), channels=channels, samplerate=sample_rate, latency=latency, dtype=dtype)
        stream.start()

        def write_audio_frames():
            # Output all frames waiting in input queue.
            for update_step in get_all_from_queue(input_queue):
                for audio_frame_list in update_step.values():
                    for frame in audio_frame_list:
                        if frame is not None:
                            stream.write(frame)

        scheduler = create_periodic_event(interval=interval, action=write_audio_frames)
        scheduler.run()
        stream.close()

###########################################################################################################
########################################      FILES     ###################################################
###########################################################################################################


class OutputVideoFile(PipelineProcess):

    def __init__(self, filename, input_stream, video_fps=30.0, dimensions=(640, 480)):
        self.filename = filename

        super().__init__(pipeline_id='OVF-' + str(filename),
                         target_function=OutputVideoFile.output_video,
                         params=(filename, video_fps, dimensions),
                         sources=[input_stream])

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    @staticmethod
    def output_video(input_queue, output_queue, filename, video_fps, dimensions):
        import cv2, time, math
        stream = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), video_fps, dimensions)

        last_frame = numpy.zeros((dimensions[1], dimensions[0], 3), dtype='uint8')
        frames_processed = 0
        start_time = time.clock()

        def write_video_frames():
            nonlocal last_frame, start_time, frames_processed

            # drop and add frames as needed to keep up with live stream
            frames_to_go = math.floor(video_fps * (time.clock() - start_time)) - frames_processed

            # write frames from input
            for update_step in get_all_from_queue(input_queue):
                assert len(update_step) == 1, 'Input too large.'
                frame_list = next(iter(update_step.values()))
                for frame in frame_list:
                    last_frame = frame if frame.shape[0:2][::-1] == dimensions else cv2.resize(frame, dimensions,
                                                                                               interpolation=cv2.INTER_AREA)
                    stream.write(last_frame)

                    # keep track of progress
                    frames_processed += 1
                    frames_to_go -= 1
                    if frames_to_go <= 0:  # break if caught up
                        return

            # pad frames if behind
            for _ in range(frames_to_go):
                stream.write(last_frame)
                frames_processed += 1

        scheduler = create_periodic_event(interval=1.0/video_fps, action=write_video_frames)
        scheduler.run()


class OutputAudioFile(PipelineProcess):

    def __init__(self, filename, input_stream, sample_rate, channels=1, interval=1/30):
        super().__init__(pipeline_id='OAF-' + str(filename),
                         target_function=OutputAudioFile.output_audio,
                         params=(filename, sample_rate, channels, interval),
                         sources=[input_stream])

    def read(self):
        raise ReadFromOutputException('Attempted read from an output pipeline function.' + str(self.__class__))

    @staticmethod
    def output_audio(input_queue, output_queue, filename, sample_rate, channels, interval):
        stream = soundfile.SoundFile(filename, mode='w', samplerate=sample_rate, channels=channels)

        def write_audio_frames():
            # collect all data in queue
            for update_step in get_all_from_queue(input_queue):
                for frame_list in update_step.values():
                    for frame in frame_list:
                        if type(frame) == numpy.ndarray:
                            stream.write(frame)

        scheduler = create_periodic_event(interval=interval, action=write_audio_frames)
        scheduler.run()

###########################################################################################################
########################################      Join Fn     #################################################
###########################################################################################################


def join_audio_and_video(audio_filename, video_filename):
    import subprocess

    cmd = 'ffmpeg -y -i ' + video_filename + ' -i ' + audio_filename + ' -shortest -async 1 -vsync 1 -codec copy output_files/output.avi'
    # flags  -codec copy
    subprocess.call(cmd, shell=True)