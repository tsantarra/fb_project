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
# http://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic/4160733#4160733


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
