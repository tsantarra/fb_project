import wave
import struct
import cv2


def get_wav_frames_from_file(filename):
    """ Returns a wav file as an array """
    wave_file = wave.open(filename, 'r')

    length = wave_file.getnframes()
    for _ in range(0, length):
        wave_data = wave_file.readframes(1)
        data = struct.unpack("<h", wave_data)
        yield int(data[0])

    wave_file.close()


# STREAMING AUDIO
# https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio-documentation
# http://people.csail.mit.edu/hubert/pyaudio/     -> see "record" example
# http://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic/4160733#4160733

# http://python-sounddevice.readthedocs.io/en/0.3.3/#device-selection

# What sounddevice is based on: https://github.com/bastibe/PySoundCard/


def get_audio_from_stream():
    """ This appear to fit the desired generator style well!"""
    import sounddevice as sd
    with sd.InputStream(device=0, channels=1, latency='low') as stream:
        while True:  # Need break case
            data, flag = stream.read(stream.read_available)
            yield data



def get_video_frames_from_file(vid_filename):
    cap = cv2.VideoCapture(vid_filename)

    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break


def get_video_frames_from_stream(device_id=0):
    cap = cv2.VideoCapture(device_id)

    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            cap.release()
            break
