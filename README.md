# SilenceRemover by IlPakoZ

*SilenceRemover* is a Python tool to remove silence from audio and video files.<br/>
To see the available options, download silenceremover.py and run the command ```python silenceremover.py --help``` on your command prompt.

## Requirements

Until a more time and memory efficient algorithm is developed, it is strongly recommended to not cut files longer than 2 hours.<br/>

This script saves temporary files on your disk, so make sure it has the right permissions to do so. For the same reason, a bit of disk space
is required for the proper functioning of the script (usually, not more than 3 GBs, but it may be higher for longer files).
RAM usage may also be high for long files.