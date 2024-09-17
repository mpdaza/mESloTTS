@echo off
setlocal enabledelayedexpansion

rem Specify input and output folders here
set "INPUT_FOLDER=data\example\mp3s"
set "OUTPUT_FOLDER=data\example\wavs"

rem Create output folder if it doesn't exist
if not exist "%OUTPUT_FOLDER%" mkdir "%OUTPUT_FOLDER%"

rem Initialize counter for sequential naming
set /a counter=1

rem Loop through all MP3 files in the input folder
for %%F in ("%INPUT_FOLDER%\*.mp3") do (
    rem Convert the file using ffmpeg
    ffmpeg -i "%%F" -acodec pcm_s16le -ar 44100 "%OUTPUT_FOLDER%\!counter!.wav"
    
    rem Increment the counter
    set /a counter+=1
)

echo Conversion complete. Check the '%OUTPUT_FOLDER%' folder for the converted files.
