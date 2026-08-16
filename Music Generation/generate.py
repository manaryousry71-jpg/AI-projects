"""
Music Generation AI
Music Generation Module

This script:
1. Loads the trained model.
2. Loads saved dictionaries.
3. Generates new notes using temperature sampling.
4. Converts generated notes to MIDI.
5. Saves the generated MIDI file.
"""

# Import Required Libraries
import os
import pickle
import random
import numpy as np
from keras.models import load_model
from music21 import note, chord, stream

# Configuration
MODEL_PATH = "model/music_lstm.keras"
DATA_FOLDER = "data"
OUTPUT_FOLDER = "generated"
OUTPUT_FILE = "generated_music_improved.mid"
SEQUENCE_LENGTH = 100
GENERATE_LENGTH = 500
TEMPERATURE = 0.8

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Load Resources
def load_resources():
    print("Loading trained model...")
    model = load_model(MODEL_PATH)
    
    print("Loading training sequences...")
    network_input = np.load(
        os.path.join(DATA_FOLDER, "network_input.npy")
    )
    
    print("Loading note dictionary...")
    with open(
        os.path.join(DATA_FOLDER, "int_to_note.pkl"), "rb"
    ) as file:
        int_to_note = pickle.load(file)
        
    return model, network_input, int_to_note


# Sample From Prediction
def sample_prediction(prediction, temperature):
    # Remove batch dimension
    prediction = prediction[0]
    
    # Avoid zero probabilities
    prediction = np.asarray(prediction).astype("float64")
    prediction = np.log(prediction + 1e-8) / temperature
    
    # Convert logits to probabilities
    exp_predictions = np.exp(prediction)
    probabilities = exp_predictions / np.sum(exp_predictions)
    
    # Randomly select a note
    index = np.random.choice(len(probabilities), p=probabilities)
    return index


# Generate Music
def generate_notes(model, network_input, int_to_note):
    # Select random starting sequence
    start_index = random.randint(0, len(network_input) - 1)
    pattern = network_input[start_index].copy()
    generated_notes = []
    
    print("=" * 50)
    print("Generating music...")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Number of notes: {GENERATE_LENGTH}")
    print("=" * 50)
    
    for step in range(GENERATE_LENGTH):
        # Reshape input
        prediction_input = np.reshape(
            pattern, (1, SEQUENCE_LENGTH, 1)
        )
        
        # Predict next note
        prediction = model.predict(prediction_input, verbose=0)
        
        # Select note using temperature sampling
        index = sample_prediction(prediction, TEMPERATURE)
        
        # Convert integer to note
        result = int_to_note[index]
        generated_notes.append(result)
        
        # Normalize predicted index
        normalized_index = index / float(len(int_to_note))
        
        # Add prediction to pattern
        pattern = np.append(
            pattern, [[normalized_index]], axis=0
        )
        
        # Keep only last SEQUENCE_LENGTH notes
        pattern = pattern[-SEQUENCE_LENGTH:]
        
        # Progress message
        if (step + 1) % 50 == 0:
            print(f"Generated {step + 1}/{GENERATE_LENGTH} notes...")
            
    return generated_notes


# Convert Notes to MIDI
def create_midi(generated_notes):
    output = stream.Stream()
    print("=" * 50)
    print("Converting generated notes to MIDI...")
    
    for pattern in generated_notes:
        pattern_str = str(pattern)

        # 1. 
        if "." in pattern_str:
            notes_in_chord = pattern_str.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                if current_note.isdigit():
                    new_note = note.Note(int(current_note))
                else:
                    new_note = note.Note(current_note)
                new_note.quarterLength = 0.5
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            new_chord.quarterLength = 0.5
            output.append(new_chord)

        # 2. (Pitch Number)
        elif pattern_str.isdigit():
            new_note = note.Note(int(pattern_str))
            new_note.quarterLength = 0.5
            output.append(new_note)

        # 3. حالة النوتة الفردية المكتوبة كاسم نوتة مثل "C4" أو "A4"
        else:
            new_note = note.Note(pattern_str)
            new_note.quarterLength = 0.5
            output.append(new_note)

    # Save MIDI
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    output.write("midi", fp=output_path)
    
    print("MIDI file saved to:")
    print(output_path)


# Main
def main():
    print("=" * 60)
    print("AI MUSIC GENERATION")
    print("=" * 60)
    
    # Load resources
    model, network_input, int_to_note = load_resources()
    print("Resources loaded successfully!")
    
    # Generate music
    generated_notes = generate_notes(model, network_input, int_to_note)
    print("=" * 50)
    print(f"Generated Notes: {len(generated_notes)}")
    
    # Convert to MIDI
    create_midi(generated_notes)
    
    print("=" * 50)
    print("Music Generation Finished Successfully!")
    print("=" * 50)


# Run Program
if __name__ == "__main__":
    main()