import wave
import struct
import cv2
import sounddevice as sd


def get_camera_list():
    id = 0
    cam_list = []

    while True:
        stream = cv2.VideoCapture(id)
        if stream.isOpened():
            stream.release()
            cam_list.append(id)
            id += 1
        else:
            break

    return cam_list


def get_wav_frames_from_file(filename):
    """ Returns a wav file as an array """
    wave_file = wave.open(filename, 'r')

    length = wave_file.getnframes()
    for _ in range(0, length):
        wave_data = wave_file.readframes(1)
        data = struct.unpack("<h", wave_data)
        yield int(data[0])

    wave_file.close()


def get_audio_from_stream():
    """ This appear to fit the desired generator style well!"""
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
            cap.release()
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
