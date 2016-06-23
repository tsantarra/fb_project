import cv2
import sounddevice
import wave
import subprocess


class Output:
    def write(self, data):
        raise NotImplementedError


class OutputVideoStream(Output):
    def __init__(self, id):
        self.id = str(id)

    def write(self, data):
        cv2.imshow(self.id, data)
        cv2.waitKey(1)


class OutputAudioStream(Output):
    def __init__(self, device_id, channels=1, samplerate=44100, latency='low' ):
        self.stream = sounddevice.OutputStream(device=device_id, channels=channels, samplerate=samplerate, latency=latency)
        self.stream.start()

    def write(self, data):
        self.stream.write(data)

    def __del__(self):
        self.stream.close()


class OutputVideoFile(Output):
    def __init__(self, filename, video_fps, dimensions):
        self.stream = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), video_fps, dimensions)

    def write(self, data):
        self.stream.write(data)

    def __del__(self):
        self.stream.release()


class OutputAudioFile(Output):
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def write(self, data):
        self.data.extend(data)

    def __del__(self):
        wave_file = wave.open(self.filename, 'w')
        wave_file.setnframes(len(self.data))
        wave_file.setframerate(44100)
        wave_file.setnchannels(len(self.data[0]))
        wave_file.writeframes(self.data)
        wave_file.close()


def join_audio_and_video(audio_filename, video_filename):
    cmd = 'ffmpeg -i ' + video_filename + ' -i ' + audio_filename + ' -codec copy -shortest output.avi'
    subprocess.call(cmd, shell=True)