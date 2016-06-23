import sounddevice
from data_sources import get_camera_list

if __name__ == '__main__':
    print('Audio devices:\n', sounddevice.query_devices())
    print('Active cameras:', get_camera_list())
