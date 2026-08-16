# Music Generation using LSTM
An AI-based music generation project that uses a Long Short-Term Memory (LSTM) neural network to learn musical note sequences from MIDI files and generate new music.

## Overview
This project demonstrates how deep learning can be applied to music generation. The model is trained on sequences of musical notes extracted from MIDI files, learns the underlying patterns, and then generates new note sequences that are converted into a MIDI file.

## Features
* Loads and preprocesses MIDI music files.
* Extracts musical note sequences.
* Trains an LSTM neural network using TensorFlow/Keras.
* Generates new music using temperature sampling.
* Converts generated notes into a MIDI file.
* Saves the generated music for playback.


## Technologies Used
* Python
* TensorFlow / Keras
* LSTM (Long Short-Term Memory)
* NumPy
* music21
* MIDI files

## How It Works
1. MIDI files are loaded from the `data/` directory.
2. Musical notes are extracted and converted into numerical representations.
3. Sequences of notes are created to train the LSTM model.
4. The trained model predicts the next note based on the previous sequence.
5. Temperature sampling is used to control the creativity of the generated music.
6. The generated notes are converted back into a MIDI file.

## Temperature Sampling
The generation process uses a temperature parameter to control randomness.
* Lower temperature produces more predictable music.
* Higher temperature produces more diverse and creative music.

The project uses a temperature value of `0.8` for balanced generation.
## How to Run
Install the required dependencies : pip install -r requirements.txt
Train the model : python train.py
Generate new music : python generate.py
The generated MIDI file will be saved in the `generated/` directory.

## Output
The model generates a new MIDI file containing an original sequence of musical notes learned from the training dataset.

Example output : generated/generated_music.mid


## Learning Objectives
* Sequence modeling with LSTM networks.
* Working with MIDI files in Python.
* Applying deep learning to generative AI.
* Understanding temperature sampling for sequence generation.

## Future Improvements
* Train on larger and more diverse music datasets.
* Experiment with GRU or Transformer-based models.
* Add support for multiple instruments.
* Generate longer and more structured musical compositions.
