import logging
import sys
import traceback
import sched
import time
import argparse  # eventual command line parsing   https://docs.python.org/3.5/library/argparse.html

from data_sources import *
import cv2


def init():
    # Will probably parse arguments or a configuration file here.
    video_stream_list = get_camera_list()
    video_sources = [VideoStream(id) for id in video_stream_list]

    audio_stream_list = []

    features = []

    return video_sources, features


def tick(sources, features):
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
    for feature in features:
        feature.update()

    # Update Output Stream

    # Testing video stream. Not permanent code.
    for id, source in enumerate(sources):
        next_bit = next(source)
        cv2.imshow(str(id), next_bit)
        cv2.waitKey(1)


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        # Initialize system sources and features calculated over sources
        sources, features = init()

        # Establish system scheduler for periodic updates.
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


        system = sched.scheduler(time.time, time.sleep)
        periodic(scheduler=system, interval=0.01, action=tick, action_args=(sources, features))

        system.run()

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
