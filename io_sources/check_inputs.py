import sounddevice
import cv2


def get_camera_list():
    """ Unfortunately, it seems that OpenCV does not provide a way to acquire a
        list of active devices. There is likely a better way than iterating through
        possible device IDs and trying each.
    """
    id = 0
    cam_list = []

    while True:
        stream = cv2.VideoCapture(id)
        if stream.isOpened():
            # Check stream features
            ret, frame = stream.read()
            height, width, layers = frame.shape
            print('Stream id:', id)
            print('\tHeight: {} Width: {} Layers: {}'.format(height, width, layers))
            print('\tStatus:', ret)

            # Release and log id
            stream.release()
            cam_list.append(id)
            id += 1
        else:
            stream.release()
            break

    return cam_list

if __name__ == '__main__':
    print('Audio devices:\n', sounddevice.query_devices())
    print('Active cameras:', get_camera_list())
