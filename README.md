# USAGE OF mESloTTS
TTS tool for male spanish voices.

1. Clone the repository by:

```sh
git git@github.com:mpdaza/mESloTTS.git
```

2. Create conda environment and activate, for this project is melotts-win in LOCAL WINDOWS

```sh
conda create -n melotts python=3.10 
conda activate melotts
```
<!-- 3. Create env and activate in the SERVER LINUX
```sh
cd MELO-OPTIMIZACION/melo/data
sudo apt install python3-venv
python3.10 -m venv melotts
source melotts/bin/activate
``` -->

3. Installing other dependences
```sh
cd MESLOTTS
pip install -e .
pip install -r requirements.txt
python setup.py install 
```

### INFERENCE
For inference it is needed: G_X.pth and config.json in the same path! In this case, the models are in the folder melo/data/example/e1/models and the outputs will be save in the melo/data/example/ex1/outputs. It is necessary to change <e1> and <G.pth>.


4. For the inference it is important to define some phrase, generator.pth and the output
```sh
cd MESLOTTS/melo
python infer.py --text "input text" -m data/example/e1/models/G.pth -o data/example/e1/outputs
```


<!-- 5. If the files use the codec MP3 it is necessary to implement this step to covert to wav, otherwise ignore it. 

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
For train it is needed: config.json and wavs in melo/data/example
10. Train the model FOR WINDOWS
```sh
train.bat 
```
For UBUNTU
```sh
./train.sh 
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
## WAND
Because of weigths and biases is implemented it it need to create an account in https://wandb.ai/home  
14. Install wandb in local and login to the platform
```sh
pip install wandb
wandb login
```
It is probably that you will need to add an ssh-key to login.

15. Create a project in the api that would have the same name as the line 37 of train.py that is in the melo folder.
```sh
    wand_project_melo = 'mi-project'
    wandb.init(project=wand_project_melo)
```

In the wand api you will see in the project the different runs you make to the project.  -->