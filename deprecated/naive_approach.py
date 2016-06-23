import wave
import struct
import cv2
import subprocess


def get_wav_frames(filename):
    """ Returns a wav file as an array """
    wave_file = wave.open(filename, 'r')

    length = wave_file.getnframes()
    frames = []
    for i in range(0, length):
        wave_data = wave_file.readframes(1)
        data = struct.unpack("<h", wave_data)
        frames += [int(data[0])]
    wave_file.close()
    return frames


def apply_max_filter_reduced_fps(wav_frames, audio_fps, video_fps):
    """ Applies a max filter to a set of wavFrames, in blocks
        of length audioFPS/videoFPS
    """
    new_frames = []
    block_length = audio_fps / video_fps
    print("Applying Filter")
    print('|' + ' ' * 50 + '|')
    print(' ', end='')
    tick = (len(wav_frames) / block_length) / 50
    i = 0
    while int((i * 1.0 / video_fps) * audio_fps) + block_length < len(wav_frames):
        if i % tick == 0:
            print('=', end='')
        start = int((i * 1.0 / video_fps) * audio_fps)
        end = int(start + block_length)
        new_frames += [max(wav_frames[start:end])]
        i += 1
    print('')
    return new_frames


def get_highlight_frames(vid_wav_pairs):
    """ Input: 
        Output: [(frame, vidFilename)] for every frame in the video

        Feels like it could be optimized to use less storage/time by not keeping
        the whole sliding window
    """
    audio_fps = 16000
    video_fps = 25

    if len(vid_wav_pairs) == 0:
        return []

    # Transform all the .wav files to arrays, and apply a sliding window max filter
    all_wav_frames = []
    for (vidFilename, wavFilename) in vid_wav_pairs:
        print("Reading " + wavFilename + '...')
        wav_frames = get_wav_frames(wavFilename)
        wav_frames = apply_max_filter_reduced_fps(wav_frames, audio_fps, video_fps)
        all_wav_frames += [(vidFilename, wav_frames)]

    # Chop to the length of the shortest wav array
    min_wav_array = min([len(x[1]) for x in all_wav_frames])
    all_wav_frames = [(x[0], x[1][:min_wav_array]) for x in all_wav_frames]

    highlight_frames = []

    # For each frame, check which audio had highest max, add that video
    for i in range(len(all_wav_frames[0][1])):
        max_vid_filename = max([(x[1][i], x[0]) for x in all_wav_frames])[1]
        highlight_frames += [(i, max_vid_filename)]

    return highlight_frames


def make_naive_video(output_filename, vid_wav_pairs):
    """ Creates an avi from a list of [(vidFilename, wavFilename)] """
    highlights = get_highlight_frames(vid_wav_pairs)
    video_fps = 25

    temp_cap = cv2.VideoCapture(vid_wav_pairs[0][0])
    ret, frame = temp_cap.read()
    height, width, layers = frame.shape

    four_cc = cv2.VideoWriter_fourcc(*'DIVX')
    output_f = cv2.VideoWriter(output_filename, four_cc, video_fps, (width, height))

    input_caps = {}
    for (vid_filename, wav_filename) in vid_wav_pairs:
        cap = cv2.VideoCapture(vid_filename)
        input_caps[vid_filename] = cap

    for (i, vid_filename) in highlights:
        ret, frame = (False, [0])
        for k in input_caps:
            if k == vid_filename:
                ret, frame = input_caps[k].read()
            else:
                _, _ = input_caps[k].read()

        output_f.write(frame)

    for k in input_caps:
        input_caps[k].release()
    output_f.release()


def make_naive_video_sliding_window(output_filename, vid_wav_pairs, window_length):
    """ Creates an avi from a list of [(vidFilename, wavFilename)] """
    highlights = get_highlight_frames(vid_wav_pairs)
    video_fps = 25

    temp_cap = cv2.VideoCapture(vid_wav_pairs[0][0])
    ret, frame = temp_cap.read()
    height, width, layers = frame.shape

    four_cc = cv2.VideoWriter_fourcc(*'DIVX')
    output_f = cv2.VideoWriter(output_filename, four_cc, video_fps, (width, height))

    input_caps = {}
    window_counts = {}
    for (vid_filename, wav_filename) in vid_wav_pairs:
        cap = cv2.VideoCapture(vid_filename)
        input_caps[vid_filename] = cap
        window_counts[vid_filename] = 0

    for (i, vid_filename) in highlights:
        ret, frame = (False, [0])
        window_counts[vid_filename] += 1
        if i > window_length:
            window_counts[highlights[i - window_length][1]] -= 1

        # Get video with highest count in the window
        vid_filename = max([(window_counts[k], k) for k in window_counts])[1]
        for k in input_caps:
            if k == vid_filename:
                ret, frame = input_caps[k].read()
            else:
                _, _ = input_caps[k].read()

        output_f.write(frame)

    for k in input_caps:
        input_caps[k].release()
    output_f.release()


def make_naive_video_sliding_window_no_thrash(output_filename, vid_wav_pairs,
                                              window_length, thrash_frames=25):
    """ Creates an avi from a list of [(vidFilename, wavFilename)] """
    highlights = get_highlight_frames(vid_wav_pairs)
    video_fps = 25

    temp_cap = cv2.VideoCapture(vid_wav_pairs[0][0])
    ret, frame = temp_cap.read()
    height, width, layers = frame.shape

    four_cc = cv2.VideoWriter_fourcc(*'DIVX')
    output_f = cv2.VideoWriter(output_filename, four_cc, video_fps, (width, height))

    input_caps = {}
    window_counts = {}
    for (vid_filename, wav_filename) in vid_wav_pairs:
        cap = cv2.VideoCapture(vid_filename)
        input_caps[vid_filename] = cap
        window_counts[vid_filename] = 0

    last_change = -2 * thrash_frames
    last_filename = ''
    for (i, vid_filename) in highlights:
        ret, frame = (False, [0])
        window_counts[vid_filename] += 1
        if i > window_length:
            window_counts[highlights[i - window_length][1]] -= 1

        # Get video with highest count in the window
        vid_filename = max([(window_counts[k], k) for k in window_counts])[1]
        if vid_filename != last_filename:
            if i - last_change > thrash_frames:
                last_change = i
                last_filename = vid_filename
            else:
                vid_filename = last_filename
        for k in input_caps:
            if k == vid_filename:
                ret, frame = input_caps[k].read()
            else:
                _, _ = input_caps[k].read()

        output_f.write(frame)

    for k in input_caps:
        input_caps[k].release()
    output_f.release()


if __name__ == '__main__':
    intermediate_filename = 'vid_no_audio.avi'
    audio_path = 'D:/"Google Drive"/amicorpus/amicorpus/IS1000a/audio/IS1000a.Array2-02.wav'

    make_naive_video(intermediate_filename, [('D:/Google Drive/amicorpus/IS1000a/video/IS1000a.Closeup1.avi',
                                              'D:/Google Drive/amicorpus/IS1000a/audio/IS1000a.Headset-0.wav'),
                                             ('D:/Google Drive/amicorpus/IS1000a/video/IS1000a.Closeup2.avi',
                                              'D:/Google Drive/amicorpus/IS1000a/audio/IS1000a.Headset-1.wav'),
                                             ('D:/Google Drive/amicorpus/IS1000a/video/IS1000a.Closeup3.avi',
                                              'D:/Google Drive/amicorpus/IS1000a/audio/IS1000a.Headset-2.wav'),
                                             ('D:/Google Drive/amicorpus/IS1000a/video/IS1000a.Closeup4.avi',
                                              'D:/Google Drive/amicorpus/IS1000a/audio/IS1000a.Headset-3.wav'),
                                             ])

    cmd = 'ffmpeg -i ' + intermediate_filename + ' -i ' + audio_path + ' -codec copy -shortest output.avi'
    subprocess.call(cmd, shell=True)
    print('Muxing Done')
