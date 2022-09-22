import numpy as np
import time
import math
import os
import subprocess
import cv2 as cv
import sys
import getopt
from scipy.io.wavfile import read, write
from tqdm import tqdm
from enum import Enum

####### GLOBAL DEFINITIONS, DO NOT EDIT! #######

video_supp_exts = [".mp4", ".mkv"]
audio_supp_exts = [".mp3", ".wav"]

INVALID_OPTION_EXIT_STATUS = 2
OUTPUT_NOT_COMPATIBLE_EXIT_STATUS = 4
NOT_IMPLEMENTED_EXIT_STATUS = 5
NOT_VALID_FORMAT_EXIT_STATUS = 6
PATH_NOT_EXISTS_STATUS = 7
INVALID_OPTION_TYPE = 8

UNKNOWN_ERROR_STATUS = 100
TEMP_FILE_ALREADY_EXISTS_STATUS = 101
STREAM_CLOSED_UNEXPECTEDLY = 102
GENERIC_EXCEPTION_STATUS = 103

WINDOW_FACTOR = 80
MARGIN = 30


class CompressionAlgo(Enum):
    LIGHT = 1
    MID = 2
    HEAVY = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

#################################################


# Converts a video or audio file from extension ext to extension '.wav'.
# For video files, the operation corresponds to extracting the audio to a .wav file.
def get_wav(name, ext):
    try:
        subprocess.check_call(
            ['ffmpeg', '-i', f'{name}{ext}', f'{name}.wav'])
    except subprocess.CalledProcessError:
        if os.path.exists(f'{name}.wav'):
            print(f"A file named {name}.wav already exists!")
            sys.exit(TEMP_FILE_ALREADY_EXISTS_STATUS)
        else:
            print(f"An unknown error occurred while extracting .wav file from {name}{ext}.")
            sys.exit(UNKNOWN_ERROR_STATUS)


# Removes silence from a video file
def cut_video(video_name, video_ext, output_name=None, silence_threshold=None, compress=None):
    video = cv.VideoCapture(f"{video_name}{video_ext}")

    # Extract the audio from the video.
    get_wav(video_name, video_ext)

    if video.isOpened():
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))  # Frame width
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # Frame height
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))  # Number of frames in the video
    else:
        print("Video streams closed unexpectedly. The program will be terminated.")
        return STREAM_CLOSED_UNEXPECTEDLY

    samplerate, data = read(f"{video_name}.wav")
    duration = len(data) / samplerate
    final_mask = get_edited_audio_matrix(data, samplerate, silence_threshold)
    final_audio = data[final_mask]
    # Duration of the video after the removal of silence
    new_duration = len(final_audio) / samplerate

    framerate = round(frame_count / duration)  # Frame rate of the video

    # To keep contains the indexes of the audio samples from the original audio data array to keep
    to_keep = np.where(final_mask == True)[0]
    # Group the samples to keep in arrays of the same or near-same length so that
    # there is a group for each frame that there will be in the final cut video.
    subd = np.array_split(to_keep, math.ceil(new_duration * framerate))

    if not output_name:
        output_name = f"{video_name}_sr.mp4"

    temp_written_video = f"{video_name}_temp.mp4"
    temp_written_audio = f"{video_name}_temp.wav"

    fourcc = -1

    if compress == CompressionAlgo.MID:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter(temp_written_video, fourcc, framerate, (width, height))

    expected_frame = 0
    increment = samplerate / framerate

    if video.isOpened() and out.isOpened():
        # Read the frames from the original video file and write on a new Stream
        # the frames that the first audio sample of each group points to.
        for group in tqdm(subd):
            while group[0] >= int(expected_frame):
                _ = video.grab()
                expected_frame += increment

            _, frame = video.retrieve()
            out.write(frame)
    else:
        print("Video streams closed unexpectedly. The program will be terminated.")
        return STREAM_CLOSED_UNEXPECTEDLY

    # Write the final audio to disk and close the read and write video streams.
    write(temp_written_audio, samplerate, final_audio)
    video.release()
    out.release()

    try:
        # Check if compression is requested and, eventually, execute it.
        if compress == CompressionAlgo.HEAVY:
            cp_st = time.time()
            subprocess.check_call(
                ['ffmpeg', '-i', temp_written_video, '-i', temp_written_audio, '-c:v', 'libx264', "-crf", "20", output_name])

            cp_et = time.time()
            print(f"\nCompression completed in {cp_et - cp_st}s!")
        else:
            subprocess.check_call(
                ['ffmpeg', '-i', temp_written_video, '-i', temp_written_audio, "-vcodec", "copy", output_name])

    except subprocess.CalledProcessError:
        if os.path.exists(output_name):
            print(f"A file named {output_name} already exists!")
            sys.exit(TEMP_FILE_ALREADY_EXISTS_STATUS)
        else:
            print(f"Unknown error while merging video and audio.")
            sys.exit(UNKNOWN_ERROR_STATUS)

    # Remove the temporary files.
    try:
        os.remove(temp_written_video)
        os.remove(temp_written_audio)
        os.remove(f"{video_name}.wav")
    except Exception as ex:
        print(ex)


# Returns an array of booleans 'final_mask' that indicates which audio samples
# from the original audio file to keep (True) or to discard (False).
def get_edited_audio_matrix(data, samplerate, silence_threshold):
    channels = len(data.shape)
    # If there are two channels, the mean of the values of the
    # two channels are used to calculate the silence threshold.
    if channels == 2:
        data_tmp = np.mean(np.abs(data), axis=1)
    else:
        data_tmp = np.abs(data)

    if silence_threshold is None:
        silence_threshold = np.mean(data_tmp) / 3

    # The number of consecutive frames of value less than the silence threshold needed
    # for that section of the audio to be considered "silent".
    consec_frames = samplerate // WINDOW_FACTOR
    mask = np.abs(data_tmp) < silence_threshold

    # To check contains the indexes of the audio samples from the original audio data array to keep
    to_check = np.where(mask == False)[0]
    final_mask = np.ones(mask.shape, dtype=bool)

    last_ind = 0
    for ind in tqdm(to_check):
        # If the number of consecutive silenced frames is greater than 'consec_frames'...
        if ind - last_ind > consec_frames:
            # Do not keep those samples, except for the first 'MARGIN' and last 'MARGIN', to make
            # the transition from silence to sound less rough.
            final_mask[last_ind + MARGIN:ind - MARGIN] = False

        # Update the last index
        last_ind = ind

    return final_mask


# Removes silence from an audio file
def cut_audio(audio_name, audio_ext, output_name=None, silence_threshold=None):
    # Convert the audio to .wav if needed.
    if audio_ext != ".wav":
        get_wav(audio_name, audio_ext)

    # Gets the sample rate of the audio file and an array containing the audio samples.
    samplerate, data = read(f"{audio_name}.wav")
    final_mask = get_edited_audio_matrix(data, samplerate, silence_threshold)

    if not output_name:
        output_name = f"{audio_name}_sr{audio_ext}"

    try:
        write(output_name, samplerate, data[final_mask])
    except Exception as ex:
        print(ex)
        sys.exit(GENERIC_EXCEPTION_STATUS)

    if audio_ext != ".wav":
        try:
            os.remove(f"{audio_name}.wav")
        except Exception as ex:
            print(ex)


# Prints the help screen
def _usage():
    print("usage: python3 [options] input_file.ext\n")
    print("Arguments:")
    print("\tinput_file                     The name of the input video or audio file to process.")
    print("\t                               Currently supported extensions are:")
    print("\t                               .mp4, mkv, .mp3, .wav")
    print("\n")
    print("Options:")
    print("\t-o, --output                   The output file name.")
    print("\t                               The file name must be inserted without extension")
    print("\t                               nor directory path. The file will be saved in the same")
    print("\t                               directory of the input file.")
    print("\t                               The default value is '{input_file}_sr.mp4'.")
    print("\t-c, --compress                 If the input file is a video file, you can select a compression")
    print("\t                               algorithm to make it smaller. The accepted value is a value")
    print("\t                               from 1 to 3. The highest the number, the smaller the final file,")
    print("\t                               but the more time it will take for the script to complete.")
    print("\t                               If a value not in the specified range is selected, the default")
    print("\t                               value will be used.")
    print("\t                               The default value is '1'.")
    print("\t-t, --threshold                Allows the user to specify a silence threshold manually")
    print("\t                               which is the values below which a sound is considered silence.")
    print("\t                               The default value is calculated by taking the mean of the absolute")
    print("\t                               values of the values read by scipy and divides the result by")
    print("\t                               a constant factor. Using this option is not recommended given")
    print("\t                               the high variability of the value between files.")
    print("\t-h, --help                     Show this help screen.")


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:c:t:l", ["help", "output=", "compress=", "threshold="])
    except getopt.GetoptError as err:
        # Print error and exit
        print(err)
        _usage()
        sys.exit(INVALID_OPTION_EXIT_STATUS)

    threshold = None
    output = None
    compress = CompressionAlgo(1)
    countOpts = 0
    for o, a in opts:
        if o in ("-h", "--help"):
            _usage()
            sys.exit()
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-c", "--compress"):
            if str.isdigit(a):
                compress = int(a)
                if CompressionAlgo.has_value(compress):
                    compress = CompressionAlgo(compress)
        elif o in ("-t", "--threshold"):
            try:
                threshold = float(a)
            except ValueError:
                print(f"Argument {a} for threshold is not a valid float.")
                sys.exit(INVALID_OPTION_TYPE)
        elif o in "-l":
            pass
        else:
            sys.exit(3)

        countOpts += 1

        if o[1] != "-" and not a:
            countOpts += 1

    if len(sys.argv[countOpts + 1:]) > 1:
        if output:
            print("Output option can be used when the input file is just one.")
            sys.exit(OUTPUT_NOT_COMPATIBLE_EXIT_STATUS)

    for par in sys.argv[countOpts + 1:]:
        if os.path.exists(par):
            file_name, file_extension = os.path.splitext(par)
            new_out = output
            if output:
                new_out = os.path.dirname(par) + '\\' + output + file_extension

            if file_extension.lower() in video_supp_exts:
                # Analyze video
                cut_video(file_name, file_extension, output_name=new_out, silence_threshold=threshold,
                          compress=compress)
            elif file_extension.lower() in audio_supp_exts:
                # Analyze audio
                cut_audio(file_name, file_extension, output_name=new_out, silence_threshold=threshold)
            elif not file_extension:
                print("Folder analysis is not yet implemented.")
                sys.exit(NOT_IMPLEMENTED_EXIT_STATUS)
            else:
                print(f"File format {file_extension} not valid.")
                sys.exit(NOT_VALID_FORMAT_EXIT_STATUS)
        else:
            print(f"The path {par} doesn't exists.")
            sys.exit(PATH_NOT_EXISTS_STATUS)
