import sched
import time


def periodic(scheduler, interval, action, action_args=(), halt_check=None):
    """ This design pattern schedules the next scheduling event, then separately executes the desired action. The
        reason for this is that we desire an arbitrarily long series of repeated loops rather than a finite series
        of events that would be scheduled all up front.
    """
    # Schedule next iteration or terminate
    if (halt_check is None) or (not halt_check()):
        scheduler.enter(delay=interval, priority=1, action=periodic,
                        argument=(scheduler, interval, action, action_args, halt_check))
    else:
        print('Ended:', action)
        return

    # Perform action with specified args
    action(*action_args)


def create_periodic_event(interval, action, action_args=(), halt_check=None):
    scheduler = sched.scheduler(time.time, time.sleep)
    periodic(scheduler=scheduler, interval=interval, action=action, action_args=action_args, halt_check=halt_check)

    return scheduler
