import cv2
import sounddevice
import soundfile
import wave
import subprocess
import struct


class Output:
    def write(self, data):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class OutputVideoStream(Output):
    def __init__(self, id):
        self.id = str(id)

    def write(self, data):
        cv2.imshow(self.id, data)
        cv2.waitKey(1)

    def close(self):
        pass


class OutputAudioStream(Output):
    def __init__(self, device_id, channels=1, samplerate=44100, latency='low' ):
        self.stream = sounddevice.OutputStream(device=device_id, channels=channels, samplerate=samplerate, latency=latency)
        self.stream.start()

    def write(self, data):
        self.stream.write(data)

    def close(self):
        self.stream.close()


class OutputVideoFile(Output):
    def __init__(self, filename, video_fps=20, dimensions=(640, 480)):
        self.stream = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), video_fps, dimensions)
        self.filename = filename

    def write(self, data):
        self.stream.write(data)

    def close(self):
        self.stream.release()


class OutputAudioFile(Output):
    def __init__(self, filename, samplerate=44100, channels=1, subtype='PCM_24'):
        self.output_stream = soundfile.SoundFile(filename, mode='w', samplerate=samplerate,
                                 channels=channels, subtype=subtype)

    def write(self, data):
        self.output_stream.write(data)

    def close(self):
        self.output_stream.close()
        return


def join_audio_and_video(audio_filename, video_filename):
    cmd = 'ffmpeg -i ' + video_filename + ' -i ' + audio_filename + ' -codec copy -shortest output.avi'
    subprocess.call(cmd, shell=True)