
A framework for automatic view switching in multi-camera remote meetings. 

Setup
=====
Each of the setup parameters can be found in the config.ini file. The system operates in two basic modes, live input and file input. 

Live Mode
---------
Device IDs for each of the cameras and microphones need to be specified in advance. Running the script check_inputs.py from the util directory will output active audio and video devices along with their device IDs. Below is an outline of the live mode parameters:

* active_camera_ids - An array of camera device IDs to be used.
* active_microphone_ids - An array of microphone device IDs to be used.
* microphone_camera_mapping - A pairing camera and microphone IDs, e.g. [ (audio_id1, video_id1), (audio_id2, video_id2)].
* audio_input_device_id - The device ID for the main audio input device, which will be recorded and also output during the live stream.

File Mode
---------
In file mode, the system will require a list of filenames for both audio and video input. 

* video_filenames - A list of input video filenames. 
* audio_filenames - A list of input audio filenames, given corresponding order to match the input video filenames.
* main_audio_file - The primary audio source filename. 

Output
---------
Regardless of input mode, all output is streamed live. Additional parameters for recording output files are outlined below:

[OUTPUT_AUDIO]
* audio_file - Boolean, indicating if a file should be recorded.
* audio_output_device_id - The device ID to which output audio should be routed for live display.
* audio_filename - The filename for the output audio file, should one be recorded. 

[OUTPUT_VIDEO]
* video_file - Boolean, indicating if a file should be recorded.
* video_filename = The filename for the output video file, should one be recorded.

