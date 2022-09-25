import numpy as np
import time
import os
import subprocess
import cv2 as cv
import sys
import getopt
from scipy.io.wavfile import read, write
from tqdm import tqdm
from enum import Enum
import decimal

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

WINDOW_FACTOR = 5
MARGIN = 100


class CompressionAlgo(Enum):
    LIGHT = 1
    MID = 2
    HEAVY = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ThresholdAlgo(Enum):
    SENSITIVE = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4

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
def cut_video(video_name, video_ext, output_name=None, method=ThresholdAlgo.MODERATE, compress=None):
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
    duration = decimal.Decimal(len(data)) / samplerate
    final_mask = get_edited_audio_matrix(data, samplerate, method)
    final_audio = data[final_mask]
    # Duration of the video after the removal of silence
    new_duration = decimal.Decimal(len(final_audio)) / samplerate
    framerate = decimal.Decimal(frame_count) / duration  # Frame rate of the video

    # To keep contains the indexes of the audio samples from the original audio data array to keep
    to_keep = np.where(final_mask == True)[0]

    if not output_name:
        output_name = f"{video_name}_sr.mp4"

    temp_written_video = f"{video_name}_temp.mp4"
    temp_written_audio = f"{video_name}_temp.wav"

    fourcc = -1

    if compress == CompressionAlgo.MID:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter(temp_written_video, fourcc, frame_count/float(duration), (width, height))
    expected_frame = 0
    increment = samplerate / framerate

    print("Processing video file...", flush=True)
    if video.isOpened() and out.isOpened():
        # Read the frames from the original video file and write on a new Stream
        # the frames that the first audio sample of each group points to.
        for index in tqdm(np.arange(0, len(to_keep), increment)):
            while to_keep[int(index)] >= expected_frame:
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
def get_edited_audio_matrix(data, samplerate, method=ThresholdAlgo.MODERATE):
    channels = len(data.shape)
    # If there are two channels, the mean of the values of the
    # two channels are used to calculate the silence threshold.
    if channels == 2:
        data_tmp = np.mean(np.abs(data), axis=1)
    else:
        data_tmp = np.abs(data)

    if method == ThresholdAlgo.SENSITIVE:
        first_threshold = np.mean(data_tmp)
        mask1 = np.abs(data_tmp) < first_threshold
        data_mask = data_tmp[mask1]
        silence_threshold = (np.mean(data_mask)) * 2
    elif method == ThresholdAlgo.WEAK:
        silence_threshold = np.mean(data_tmp) / 2
    elif method == ThresholdAlgo.MODERATE:
        first_threshold = np.mean(data_tmp)
        mask1 = np.abs(data_tmp) < first_threshold
        data_mask = data_tmp[mask1]
        silence_threshold = (np.mean(data_mask)+first_threshold) / 2
    elif method == ThresholdAlgo.STRONG:
        silence_threshold = np.mean(data_tmp)

    # The number of consecutive frames of value less than the silence threshold needed
    # for that section of the audio to be considered "silent".
    consec_frames = samplerate // WINDOW_FACTOR
    mask = np.abs(data_tmp) < silence_threshold

    # To check contains the indexes of the audio samples from the original audio data array to keep
    to_check = np.where(mask == False)[0]
    final_mask = np.ones(mask.shape, dtype=bool)
    last_ind = 0
    print("Cutting audio file...", flush=True)
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
def cut_audio(audio_name, audio_ext, output_name=None, method=ThresholdAlgo.MODERATE):
    # Convert the audio to .wav if needed.
    if audio_ext != ".wav":
        get_wav(audio_name, audio_ext)

    # Gets the sample rate of the audio file and an array containing the audio samples.
    samplerate, data = read(f"{audio_name}.wav")
    final_mask = get_edited_audio_matrix(data, samplerate, method=ThresholdAlgo.MODERATE)

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
    print("\t                               If a value not in the specified range is inserted, the default")
    print("\t                               value will be used.")
    print("\t                               WARNING: some video files may benefit better size loss with")
    print("\t                               method 1 than method 2. This depends by the specific compression")
    print("\t                               mechanisms of each algorithm: keep in mind that those numbers are")
    print("\t                               only indicative and results may vary.")
    print("\t                               The default value is '2'.")
    print("\t-m, --method                   The threshold algorithm used to calculate the silence threshold.")
    print("\t                               The silence threshold is the amplitude value under which a sample")
    print("\t                               is detected as silent. This value is automatically calculated")
    print("\t                               by the script based on the audio track of the input and the algorithm")
    print("\t                               selected. There are four algorithms that can be used to calculate")
    print("\t                               this value. The lowest the value, the more probable it is to")
    print("\t                               capture quiet noises or voices in the background.")
    print("\t                               To capture quiet voices in the background, a lower value")
    print("\t                               is recommended. Otherwise, a higher value is recommended.")
    print("\t                               If a value not in the specified range is inserted, the default")
    print("\t                               value will be used.")
    print("\t                               The default value is '2'.")
    print("\t-h, --help                     Show this help screen.")


# Main method, reads the arguments/options and prepares the program
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:c:m:", ["help", "output=", "compress=", "method="])
    except getopt.GetoptError as err:
        # Print error and exit
        print(err)
        _usage()
        sys.exit(INVALID_OPTION_EXIT_STATUS)

    output = None
    method = ThresholdAlgo(2)
    compress = CompressionAlgo(2)
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
        elif o in ("-m", "--method"):
            if str.isdigit(a):
                method = int(a)
                if ThresholdAlgo.has_value(method):
                    method = ThresholdAlgo(method)
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
                cut_video(file_name, file_extension, output_name=new_out, method=method,
                          compress=compress)
            elif file_extension.lower() in audio_supp_exts:
                # Analyze audio
                cut_audio(file_name, file_extension, output_name=new_out, method=method)
            elif not file_extension:
                print("Folder analysis is not yet implemented.")
                sys.exit(NOT_IMPLEMENTED_EXIT_STATUS)
            else:
                print(f"File format {file_extension} not valid.")
                sys.exit(NOT_VALID_FORMAT_EXIT_STATUS)
        else:
            print(f"The path {par} doesn't exists.")
            sys.exit(PATH_NOT_EXISTS_STATUS)


if __name__ == '__main__':
    main()
