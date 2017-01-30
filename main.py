import ast
import configparser
import logging
import sys
import traceback

from collections import namedtuple
from functools import partial

import cv2
from numpy import zeros

from features.video_movement_feature import VideoMovementFeature
from io_sources.data_output import OutputVideoStream, OutputAudioStream, OutputAudioFile, OutputVideoFile, join_audio_and_video
from io_sources.data_sources import InputVideoStream, InputAudioStream, InputVideoFile, InputAudioFile
from util.schedule import create_periodic_event

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
        input_audio = [InputAudioStream(id, sample_rate=global_sample_rate, dtype=global_dtype) for id in params['AUDIO']['active_microphone_ids']]
        main_audio_input = [stream for stream in input_audio
                            if stream.id == params['OUTPUT_AUDIO']['audio_input_device_id']][0]
        input_video = [InputVideoStream(id) for id in params['VIDEO']['active_camera_ids']]

    else:
        input_audio = [InputAudioFile(filename) for filename in params['AUDIO']['audio_filenames']]
        main_audio_input = InputAudioFile(filename=params['AUDIO']['main_audio_file'])
        input_video = [InputVideoFile(filename) for filename in params['VIDEO']['video_filenames']]

    inputs = InputMediaStreams(audio=input_audio, video=input_video, main_audio=main_audio_input)

    # Features for selecting a stream
    features = [VideoMovementFeature(inputs.video),
                #AudioFeature(zip(inputs.audio, inputs.video)),
                ]

    # Output streams
    output_audio_streams = [OutputAudioStream(device_id=params['OUTPUT_AUDIO']['audio_output_device_id'],
                                              input_stream=main_audio_input, sample_rate=global_sample_rate,
                                              dtype=global_dtype)]

    output_video_streams = [OutputVideoStream(stream_id=input_stream.id, input_stream=input_stream)
                            for input_stream in input_video]

    if params['OUTPUT_AUDIO']['audio_file']:
        output_audio_streams.append(OutputAudioFile(filename=params['OUTPUT_AUDIO']['audio_filename'],
                                                    input_stream=main_audio_input,
                                                    sample_rate=global_sample_rate))

    if params['OUTPUT_VIDEO']['video_file']:
        output_video_streams.append(OutputVideoFile(filename=params['OUTPUT_VIDEO']['video_filename'],
                                                    input_stream=input_video[0]))

    outputs = OutputMediaStreams(audio=output_audio_streams, video=output_video_streams)

    # start everything
    for process in set(inputs.audio + inputs.video + [inputs.main_audio] + features + outputs.audio + outputs.video):
        process.start()

    return inputs, features, outputs, params


def tick(sources, features, output_streams):
    """
        The meat of the system runs via this tick function. All sources and feature calculators should update (hopefully
        keeping in sync. The computation of selecting a primary stream then occurs. It is yet unclear if output writing
        should occur here or as part of a separate function.
    """

    # Update Sources
    for source in set(sources.audio + sources.video + [sources.main_audio]):
        source.update()

    for output in set(output_streams.audio + output_streams.video):
        output.update()

    return


def halt_criteria(processes):
    """ This function provides the necessary check for terminating the system loop. """
    blank_image = zeros((30, 30, 3))
    cv2.imshow('Exit', blank_image)

    end = cv2.waitKey(10) == 27
    if end:
        for process in processes:
            process.close()

    return end


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        # Initialize system sources and features calculated over sources
        sources, features, output_streams, params = init()
        processes = set(sources.audio + sources.video + features + output_streams.audio + output_streams.video)

        # Initialize scheduler and set to repeat calls indefinitely
        system = create_periodic_event(interval=1/30,
                                       action=tick,
                                       action_args=(sources, features, output_streams),
                                       halt_check=partial(halt_criteria, processes))

        # Execute
        system.run()

        # Kill windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Create mixed audio/video file
        if params['OUTPUT_AUDIO']['audio_file'] and params['OUTPUT_VIDEO']['video_file']:
            join_audio_and_video(params['OUTPUT_AUDIO']['audio_filename'], params['OUTPUT_VIDEO']['video_filename'])

        print('Exit.')

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
