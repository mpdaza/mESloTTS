# USAGE OF MELOTTS

1. Clone the repository by:

```sh
git clone git@github.com:natlamir/MeloTTS-Windows.git
```

2. Create conda environment and activate, for this project is melotts-win in LOCAL WINDOWS

```sh
conda create -n <melotts-win/> python=3.10 
conda activate melotts-win
```
3. Create env and activate in the SERVER LINUX
```sh
cd Melotts-windows/melo/data
sudo apt install python3-venv
python3 -m venv melotts
source melotts/bin/activate
```

4. Installing other dependences
```sh
cd melotts-windows
pip install -e .
pip install pydub
python -m unidic download
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python setup.py install 
```

5. If the files use the codec MP3 it is necessary to implement this step to covert to wav, otherwise ignore it. 

FOR WINDOWS
```sh
cd melo
ConvertMp3toWav.bat 
./CovertMp3toWav.sh 
```

FOR UBUNTU
```sh
cd melo
./CovertMp3toWav.sh 
```
6. It is also necessary to resample to 44100Hz the frequency.
```sh
python resample.py
```

7. Transcript the wavs using the whisper "large-v3", it is necessary to change the model and to change the language, input and output folder if it is needed. 
```sh
python transcript.py
```

8. Preprocess the metadata generated from the whisper
```sh
python preprocess_text.py --metadata data/example/metadata.list
```

9. Change the config file, for both: n_speakers=256 and num_languages=10

- for the TRAIN use the config of the GitHub MeloTTS-Windows
- for the INFERENCE use the MeloTTS-Spanish GitHub but change the symbols for the MeloTTS-Windows one

### TRAIN
10. Train the model FOR WINDOWS
```sh
train.bat 
```
For UBUNTU
```sh
./train.sh 
```

### INFERENCE
11. For the inference it is important to define some phrase, generator.pth and the output
```sh
python infer.py --text <phrase> -m <root/.../G.pth> -o <root/.../output>
```

12. Using only the API creating one dir: CUSTOM with 2 files, the g.pth and the config.json
```sh
melo-ui
```

13. Using tensorboard
```sh
tensorboard --logdir=<root/.../config.json> 
python -m tensorboard.main --logdir=<root/.../model>
```
