import ast
import configparser
import logging
import sys
import traceback

from collections import namedtuple
from functools import partial

import cv2
from numpy import zeros

from features.audio_feature import AudioFeature
from features.video_movement_feature import VideoMovementFeature
from io_sources.data_output import OutputVideoStream, OutputAudioStream, OutputAudioFile, OutputVideoFile, \
    join_audio_and_video, OutputTiledVideoStream
from io_sources.data_sources import InputVideoStream, InputAudioStream, InputVideoFile, InputAudioFile
from util.distribution import Distribution
from util.schedule import create_periodic_event
from util.stream_selector import StreamSelector

InputMediaStreams = namedtuple("InputMediaStreams", ["audio", "video", "main_audio"])
OutputMediaStreams = namedtuple("OutputMediaStreams", ["audio", "video", "main_video"])

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
    parameters = parse_config_settings()

    # Streams of input data
    if parameters['MODE']['live_mode']:
        input_audio = [InputAudioStream(id, sample_rate=global_sample_rate, dtype=global_dtype)
                       for id in parameters['LIVE']['active_microphone_ids']]
        main_audio_input = [stream for stream in input_audio
                            if stream.source_id == parameters['LIVE']['audio_input_device_id']][0]
        input_video = [InputVideoStream(id) for id in parameters['LIVE']['active_camera_ids']]

        audio_video_pairs = {audio.id: video.id for audio, video in zip(input_audio, input_video)}

    else:
        input_audio = [InputAudioFile(filename) for filename in parameters['FILES']['audio_filenames']]
        main_audio_input = InputAudioFile(filename=parameters['FILES']['main_audio_file'])
        input_video = [InputVideoFile(filename) for filename in parameters['FILES']['video_filenames']]

        audio_video_pairs = {audio: video for audio, video in zip([file_stream.id for file_stream in input_audio],
                                                                  [file_stream.id for file_stream in input_video])}

    inputs = InputMediaStreams(audio=input_audio, video=input_video, main_audio=[main_audio_input])

    # Output streams
    output_audio_streams = [OutputAudioStream(device_id=parameters['OUTPUT_AUDIO']['audio_output_device_id'],
                                              input_stream=main_audio_input, sample_rate=global_sample_rate,
                                              dtype=global_dtype)]

    #output_video_streams = [OutputVideoStream(stream_id=input_stream.id, input_stream=input_stream)
    #                        for input_stream in input_video]
    output_video_streams = [OutputTiledVideoStream(stream_id='Input Streams', inputs=input_video)]

    main_video_outputs = [OutputVideoStream(stream_id="Main Output", input_stream=input_video[0])]

    # Output files
    if parameters['OUTPUT_AUDIO']['audio_file']:
        output_audio_streams.append(OutputAudioFile(filename=parameters['OUTPUT_AUDIO']['audio_filename'],
                                                    input_stream=main_audio_input,
                                                    sample_rate=global_sample_rate))

    if parameters['OUTPUT_VIDEO']['video_file']:
        output_video_file = OutputVideoFile(filename=parameters['OUTPUT_VIDEO']['video_filename'],
                                                    input_stream=input_video[0])
        output_video_streams.append(output_video_file)
        main_video_outputs.append(output_video_file)

    outputs = OutputMediaStreams(audio=output_audio_streams, video=output_video_streams,
                                 main_video=main_video_outputs)

    # Features for selecting a stream
    movement_feature = VideoMovementFeature(feature_id='F-Movement', video_sources=inputs.video)

    audio_feature = AudioFeature(feature_id='F-Audio', audio_sources=inputs.audio,
                                 audio_video_pair_map=audio_video_pairs)
    weighted_feature_distribution = Distribution({movement_feature: 0.7, audio_feature: 0.3})

    # Return StreamSelector and params
    return StreamSelector(inputs, weighted_feature_distribution, outputs), parameters


def halt_check(selector):
    """ This function provides the necessary check for terminating the system loop. """
    # display blank image
    cv2.imshow('Exit', zeros((30, 30, 3)))

    # Check for end key press (esc)
    end = (cv2.waitKey(10) == 27)
    if end:
        selector.close()

    return end


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        # Initialize system sources and features calculated over sources
        stream_selector, params = init()

        # Initialize scheduler and set to repeat update calls indefinitely
        system = create_periodic_event(interval=1 / 30,
                                       action=stream_selector.update,
                                       action_args=(),
                                       halt_check=partial(halt_check, stream_selector))

        # Execute
        system.run()

        # Kill windows
        cv2.destroyAllWindows()
        cv2.waitKey(0)

        # Create mixed audio/video file
        if params['OUTPUT_AUDIO']['audio_file'] and params['OUTPUT_VIDEO']['video_file']:
            join_audio_and_video(params['OUTPUT_AUDIO']['audio_filename'], params['OUTPUT_VIDEO']['video_filename'])

        print('Exit.')

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
