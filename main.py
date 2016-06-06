import logging
import sys
import traceback
import sched
import time
import argparse  # eventual command line parsing   https://docs.python.org/3.5/library/argparse.html

from data_sources import get_video_frames_from_stream
import cv2


def init():
    # Will probably parse arguments or a configuration file here.
    sources = [get_video_frames_from_stream()]

    # Establish system scheduler for periodic updates.
    def periodic(scheduler, interval, action, action_args=()):
        """ This design pattern schedules the next scheduling event, then separately executes the desired action. The
            reason for this is that we desire an arbitrarily long series of repeated loops rather than a finite series
            of events that would be scheduled all up front.
        """
        # Schedule next action
        scheduler.enter(delay=interval, priority=1, action=periodic, argument=(scheduler, interval, action, action_args))

        # Perform action with specified args
        action(*action_args)

    system_scheduler = sched.scheduler(time.time, time.sleep)
    periodic(scheduler=system_scheduler, interval=0.01, action=tick, action_args=(sources,))
    return system_scheduler


def tick(sources):
    """
        The meat of the system runs via this tick function. All sources and feature calculators should update (hopefully
        keeping in sync. The computation of selecting a primary stream then occurs. It is yet unclear if output writing
        should occur here or as part of a separate function.
    """
    logging.debug('Tick at time ' + str(time.time()))

    # Testing video stream. Not permanent code.
    for source in sources:
        next_bit = next(source)
        cv2.imshow('f', next_bit)
        cv2.waitKey(1)


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        system = init()
        system.run()

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
