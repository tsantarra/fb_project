import ast
import configparser
import logging
import sys
import time
import traceback
import cv2

from features.audio_feature import AudioFeature
from features.distribution import Distribution
from features.video_movement_feature import VideoMovementFeature
from io_sources.data_sources import VideoStream, AudioStream, VideoFile, AudioFile
from io_sources.data_output import OutputVideoStream, OutputAudioStream, OutputAudioFile, OutputVideoFile, join_audio_and_video
from schedule import create_periodic_event

from collections import namedtuple
InputMediaStreams = namedtuple("InputMediaStreams", ["audio", "video", "main_audio"])
OutputMediaStreams = namedtuple("OutputMediaStreams", ["audio", "video"])

global_sample_rate = 16000  # 44100 16000
global_dtype = 'Int16'  # float32  Int16


def parse_config_settings():
    """
    Reads in config.ini file with parameters for the system. Uses ast module to parse config vars as literal Python.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config._sections

    for section_name, section in config.items():
        for key, val in section.items():
            section[key] = ast.literal_eval(val)

    return config


def init():
    """
    Initializes system using parameters read from config file.
    """
    params = parse_config_settings()

    # Streams of data
    if params['MODE']['live_mode']:
        input_audio = [AudioStream(id, sample_rate=global_sample_rate, dtype=global_dtype) for id in params['AUDIO']['active_microphone_ids']]
        main_audio_input = [stream for stream in input_audio
                            if stream.id == params['OUTPUT_AUDIO']['audio_input_device_id']][0]

        input_video = [VideoStream(id) for id in params['VIDEO']['active_camera_ids']]

    else:
        input_audio = [AudioFile(filename) for filename in params['AUDIO']['audio_filenames']]
        main_audio_input = AudioFile(filename=params['AUDIO']['main_audio_file'])

        input_video = [VideoFile(filename) for filename in params['VIDEO']['video_filenames']]

    inputs = InputMediaStreams(audio=input_audio, video=input_video, main_audio=main_audio_input)

    # Features for selecting a stream
    features = [VideoMovementFeature(inputs.video),
                #AudioFeature(zip(inputs.audio, inputs.video)),
                ]

    # Output streams
    output_audio_streams = [OutputAudioStream(device_id=params['OUTPUT_AUDIO']['audio_output_device_id'], sample_rate=global_sample_rate, dtype=global_dtype)]
    output_video_streams = [OutputVideoStream(stream_name='Output')]

    if params['OUTPUT_AUDIO']['audio_file']:
        output_audio_streams.append(OutputAudioFile(filename=params['OUTPUT_AUDIO']['audio_filename'], sample_rate=global_sample_rate))

    if params['OUTPUT_VIDEO']['video_file']:
        output_video_streams.append(OutputVideoFile(filename=params['OUTPUT_VIDEO']['video_filename']))

    outputs = OutputMediaStreams(audio=output_audio_streams, video=output_video_streams)

    return inputs, features, outputs, params


def tick(sources, features, output_streams):
    """
        The meat of the system runs via this tick function. All sources and feature calculators should update (hopefully
        keeping in sync. The computation of selecting a primary stream then occurs. It is yet unclear if output writing
        should occur here or as part of a separate function.
    """
    logging.debug('Tick at time ' + str(time.time()))

    # Update Sources
    for source in set(sources.audio + sources.video + [sources.main_audio]):
        source.update()

    # update main audio source (NOTE: FIX THIS, AS SOMETIMES IT'S IN SOURCES LIST, AND SOMETIMES IT'S NOT)
    #sources.main_audio.update()

    # Update Features
    votes = []
    for feature in features:
        feature.update()
        votes.append(feature.weight_sources())

    # Vote tally
    keys = votes[0].keys()
    result = Distribution({key: 0.0 for key in keys})
    for vote in votes:
        result = result.add_distribution(vote)

    logging.debug('Vote result:' + str(result))

    # Update Output Stream
    video_source = max(result, key=lambda k: result[k])

    audio_data = sources.main_audio.read()
    for output_audio in output_streams.audio:
        output_audio.write(audio_data)

    video_data = video_source.read()
    for output_video in output_streams.video:
        output_video.write(video_data)
        
    # Testing video stream. Not permanent code.
    for i, source in enumerate(sources.video):
        if source.status:
            cv2.moveWindow(str(i), i * 640, 0)
            cv2.imshow(str(i), source.read())


def halt_criteria():
    """ This function provides the necessary check for terminating the system loop. """
    return cv2.waitKey(10) == 27


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        # Initialize system sources and features calculated over sources
        sources, features, output_streams, params = init()

        # Initialize scheduler and set to repeat calls indefinitely
        system = create_periodic_event(interval=0.001, action=tick,
                                       action_args=(sources, features, output_streams), halt_check=halt_criteria)

        # Execute
        system.run()

        # Kill windows
        cv2.destroyAllWindows()

        # Close output streams
        for output in output_streams.audio + output_streams.video:
            output.close()

        # Create mixed audio/video file
        if params['OUTPUT_AUDIO']['audio_file'] and params['OUTPUT_VIDEO']['video_file']:
            join_audio_and_video(params['OUTPUT_AUDIO']['audio_filename'], params['OUTPUT_VIDEO']['video_filename'])

        print('Exit.')

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
