import ast
import configparser
import logging
import sys
import time
import traceback
import cv2
import subprocess

from features.audio_feature import AudioFeature
from features.test_feature import TestFeature
from features.distribution import Distribution
from features.video_movement_feature import VideoMovementFeature
from io_sources.data_sources import VideoStream, AudioStream, VideoFile
from io_sources.output import OutputVideoStream, OutputAudioStream, OutputAudioFile, OutputVideoFile, join_audio_and_video
from schedule import create_periodic_event

from collections import namedtuple
InputMediaStreams = namedtuple("InputMediaStreams", ["audio", "video", "main_audio"])
OutputMediaStreams = namedtuple("OutputMediaStreams", ["audio", "video"])


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
    input_audio = [AudioStream(id) for id in params['AUDIO']['active_microphone_ids']]
    input_video = [VideoStream(id) for id in params['VIDEO']['active_camera_ids']]
    main_audio_input = [stream for stream in input_audio
                        if stream.id == params['OUTPUT_AUDIO']['audio_input_device_id']][0]
    inputs = InputMediaStreams(audio=input_audio, video=input_video, main_audio=main_audio_input)

    # Features for selecting a stream
    features = [#TestFeature(zip(inputs.audio, inputs.video)),
                VideoMovementFeature(zip(inputs.audio, inputs.video)),
                #AudioFeature(zip(inputs.audio, inputs.video)),
                ]

    # Output streams
    output_audio_streams = [OutputAudioStream(device_id=params['OUTPUT_AUDIO']['audio_output_device_id'])]
    output_video_streams = [OutputVideoStream(device_id='Output')]

    if params['OUTPUT_AUDIO']['audio_file']:
        output_audio_streams.append(OutputAudioFile(params['OUTPUT_AUDIO']['audio_filename']))

    if params['OUTPUT_VIDEO']['video_file']:
        output_video_streams.append(OutputVideoFile(params['OUTPUT_VIDEO']['video_filename']))

    outputs = OutputMediaStreams(output_audio_streams, output_video_streams)

    return inputs, features, outputs, params


def tick(sources, features, output_streams):
    """
        The meat of the system runs via this tick function. All sources and feature calculators should update (hopefully
        keeping in sync. The computation of selecting a primary stream then occurs. It is yet unclear if output writing
        should occur here or as part of a separate function.
    """
    logging.debug('Tick at time ' + str(time.time()))

    # Update Sources
    for source in sources.audio + sources.video:
        source.update()

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
    _, video_source = max(result, key=lambda k: result[k])

    for output_audio in output_streams.audio:
        output_audio.write(sources.main_audio.read())

    for output_video in output_streams.video:
        output_video.write(video_source.read())

    # Testing video stream. Not permanent code.
    for i, source in enumerate(sources.video):
        if source.status:
            cv2.moveWindow(str(i), i*640, 0)
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
        system = create_periodic_event(interval=0.01, action=tick,
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
