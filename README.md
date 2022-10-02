# SilenceRemover by IlPakoZ

_**SilenceRemover**_ is a Python tool to remove silence from audio and video files.<br/>
To see the available options, download silenceremover.py and run the command ```python silenceremover.py --help``` on your terminal.

## Requirements

You need to have the **ffmpeg** package installed on the system for the script to work.<br/>
To check if the ffmpeg package has been correctly installed on your computer, execute the command ```ffmpeg -h``` on your terminal.<br/>
If you use a Windows operating system, make sure you add the ffmpeg\bin folder to your PATH environment variable or the script will
not be able to run the ffmpeg process.

Until a more time and memory efficient algorithm is developed, it is strongly recommended to not cut files longer than 2 hours.<br/>

This script saves temporary files on your disk, so make sure it has the right permissions to do so. For the same reason, a bit of disk space
is required for the proper functioning of the script (usually, not more than 5 GBs, but it may be higher for longer files).
RAM usage may also be high for long files.