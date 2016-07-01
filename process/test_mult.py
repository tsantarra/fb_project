import cv2
import sounddevice
import numpy

from multiprocessing import Process, Queue
from schedule import create_periodic_event


def stream_audio(id, queue):
    stream = sounddevice.InputStream(device=id, channels=1, latency='low')
    stream.start()

    def grab_audio_frames(queue, stream):
        frame, flag = stream.read(stream.read_available)
        queue.put((frame, flag, stream.active))

    scheduler = create_periodic_event(interval=0.001, action=grab_audio_frames, action_args=(queue, stream))
    scheduler.run()


class AudioStream:

    def __init__(self, id):
        self.id = id
        self.last_frame = []
        self.status = True
        self.flag = None

        self.queue = Queue()
        self.process = Process(target=stream_audio, args=(id, self.queue))
        self.process.start()

    def update(self):
        frame = None
        while not self.queue.empty():
            new_frame, self.flag, self.status = self.queue.get()

            if frame is None:
                frame = new_frame
            else:
                frame = numpy.append(frame, new_frame)

            self.last_frame = frame

    def read(self):
        return self.last_frame

    def __del__(self):
        self.process.terminate()


class OutputAudioStream:
    def __init__(self, device_id, channels=1, samplerate=44100, latency='low'):
        self.stream = sounddevice.OutputStream(device=device_id, channels=channels, samplerate=samplerate,
                                               latency=latency)
        self.stream.start()

    def write(self, data):
        self.stream.write(data)

    def close(self):
        self.stream.close()


def stream_video(vid_id, queue):
    stream = cv2.VideoCapture(vid_id)

    def grab_video_frame(queue, stream):
        status, frame = stream.read()
        cv2.waitKey(1)
        queue.put((status, frame))

    scheduler = create_periodic_event(interval=0.05, action=grab_video_frame, action_args=(queue, stream))
    scheduler.run()


class VideoStream:

    def __init__(self, id, interval):
        self.status = False
        self.last_frame = None
        self.id = id

        self.queue = Queue(maxsize=1)
        self.process = Process(target=stream_video, args=(id, self.queue))
        self.process.start()

    def update(self):
        if not self.queue.empty():
            self.status, self.last_frame = self.queue.get()

    def read(self):
        return self.last_frame

    def __del__(self):
        self.process.terminate()




if __name__ == '__main__':
    inputs = [VideoStream(id, 0.05) for id in [0, 1]]

    inputs = [AudioStream(1)]
    output = OutputAudioStream(4)

    def update(inputs):
        for stream in inputs:
            stream.update()

        input = inputs[0]
        read = input.read()
        if len(read):
            output.write(read)



    system = create_periodic_event(interval=0.01, action=update, action_args=(inputs,))

    system.run()
