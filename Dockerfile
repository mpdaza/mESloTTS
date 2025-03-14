FROM python:3.10.0-slim

RUN mkdir /app

RUN mkdir /models

RUN mkdir /output

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN python setup.py install
RUN pip install -e .
# RUN python -m unidic download

# WORKDIR /app/melo

# ENV TEXT="Esto es una prueba del funcionamiento del tts"
# ENV MODEL="data/weights/G_1500.pth"
# ENV OUTPUT="data/weights/outputs"


# CMD ["python", "infer.py", "--text \"$TEXT"\", "-m \"$MODEL"\"," -o \"$OUTPUT"\"]

ENTRYPOINT [ "entrypoint.sh" ]