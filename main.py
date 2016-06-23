import ast
import configparser
import logging
import sched
import sys
import time
import traceback
import cv2

from features.audio_feature import AudioFeature
from features.distribution import Distribution
from features.video_movement_feature import VideoMovementFeature
from io_sources.data_sources import VideoStream, AudioStream, VideoFile
from io_sources.output import OutputVideoStream, OutputAudioStream


def parse_config_settings():
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config._sections

    for section_name, section in config.items():
        for key, val in section.items():
            section[key] = ast.literal_eval(val)

    return config


def init(params):
    # Will probably parse arguments or a configuration file here.
    video_stream_list = params['VIDEO']['active_camera_ids']
    video_sources = [VideoStream(id) for id in video_stream_list]

    audio_stream_list = params['AUDIO']['active_microphone_ids']
    audio_sources = [AudioStream(id) for id in audio_stream_list]

    features = [VideoMovementFeature(zip(audio_sources, video_sources)), AudioFeature(zip(audio_sources, video_sources))]

    return video_sources+audio_sources, features


def tick(sources, features, output_streams):
    """
        The meat of the system runs via this tick function. All sources and feature calculators should update (hopefully
        keeping in sync. The computation of selecting a primary stream then occurs. It is yet unclear if output writing
        should occur here or as part of a separate function.
    """
    logging.debug('Tick at time ' + str(time.time()))

    # Update Sources
    for source in sources:
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
    audio_output, video_output = output_streams
    audio_source, video_source = max(result, key=lambda k: result[k])
    audio_output.write(audio_source.read())
    video_output.write(video_source.read())

    # Testing video stream. Not permanent code.
    for id, source in enumerate(sources):
        if type(source) == VideoStream or type(source) == VideoFile:
            cv2.imshow(str(id), source.read())
            cv2.waitKey(1)


def periodic(scheduler, interval, action, action_args=()):
    """ This design pattern schedules the next scheduling event, then separately executes the desired action. The
        reason for this is that we desire an arbitrarily long series of repeated loops rather than a finite series
        of events that would be scheduled all up front.
    """
    # Schedule next action
    scheduler.enter(delay=interval, priority=1, action=periodic,
                    argument=(scheduler, interval, action, action_args))

    # Perform action with specified args
    action(*action_args)


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        # Load machine-specific config details
        params = parse_config_settings()

        # Initialize system sources and features calculated over sources
        sources, features = init(params)

        # Output streams
        output_audio_stream = OutputAudioStream(device_id=params['OUTPUT_AUDIO']['audio_stream_device_id'])
        output_video_stream = OutputVideoStream(id='Output')
        output_streams = [output_audio_stream, output_video_stream]

        # Initialize scheduler and set to repeat calls indefinitely
        system = sched.scheduler(time.time, time.sleep)
        periodic(scheduler=system, interval=0.0001, action=tick, action_args=(sources, features, output_streams))

        # Execute
        system.run()
        output_audio_stream.close()

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
