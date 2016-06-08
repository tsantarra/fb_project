#!/usr/bin/env python3

"""
Example from: http://python-sounddevice.readthedocs.io/en/0.3.3/examples.html

"""

import argparse


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
parser.add_argument('-d', '--device', type=int_or_str, help='input device (numeric ID or substring)')
parser.add_argument('-w', '--window', type=float, default=200, metavar='DURATION', help='visible time slot (default: %(default)s ms)')
parser.add_argument('-i', '--interval', type=float, default=30, help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument('-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument('-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument('-n', '--downsample', type=int, default=1, metavar='N', help='display every Nth sample (default: %(default)s)')
parser.add_argument('channels', type=int, default=[1], nargs='*', metavar='CHANNEL', help='input channels to plot (default: the first)')
args = parser.parse_args()

if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')


status_flag = True


def old_yield_func():
    """ This appear to fit the desired generator style well!"""
    import sounddevice as sd

    stream = sd.InputStream(device=args.device, channels=max(args.channels),
                            samplerate=args.samplerate, latency='low')

    stream.start()

    while status_flag:
        data, flag = stream.read(stream.read_available)
        yield data

    stream.stop()


def yield_func():
    """ This appear to fit the desired generator style well!"""
    import sounddevice as sd

    with sd.InputStream(device=args.device, channels=max(args.channels),
                            samplerate=args.samplerate, latency='low') as stream:

        while status_flag:
            data, flag = stream.read(stream.read_available)
            yield data


if __name__ == '__main__':
    import sounddevice as sd

    with sd.OutputStream(device=8, channels=max(args.channels),
                                   samplerate=args.samplerate, latency='low') as output_stream:
        for i, block in enumerate(yield_func()):
            if i > 100000:
                status_flag = False
            output_stream.write(block)
            print(i)
