"""
This script:

1. Reads MIDI files.
2. Extracts notes and chords.
3. Encodes musical symbols.
4. Creates training sequences.
5. Saves processed data.
"""

# =============================
# Import Required Libraries
# =============================

import os
import pickle
import numpy as np

from music21 import converter
from music21 import note
from music21 import chord


# =============================
# Configuration
# =============================

DATASET_PATH = "dataset/classical"
DATA_FOLDER = "data"

SEQUENCE_LENGTH = 100


# =============================
# Create Data Folder
# =============================

os.makedirs(DATA_FOLDER, exist_ok=True)


# =============================
# Read MIDI Files
# =============================

def extract_notes(dataset_path):
    """
    Read all MIDI files and extract notes and chords.
    """

    notes = []

    for file_name in os.listdir(dataset_path):

        if not file_name.endswith(".mid"):
            continue

        file_path = os.path.join(
            dataset_path,
            file_name
        )

        print(f"Reading: {file_name}")

        midi = converter.parse(file_path)

        for element in midi.flatten().notes:

            # Single note
            if isinstance(element, note.Note):

                notes.append(
                    str(element.pitch)
                )

            # Chord
            elif isinstance(element, chord.Chord):

                notes.append(
                    ".".join(
                        str(n)
                        for n in element.normalOrder
                    )
                )

    return notes


# =============================
# Encode Notes
# =============================

def encode_notes(notes):
    """
    Convert musical symbols into integers.
    """

    unique_notes = sorted(
        set(notes)
    )

    note_to_int = {
        musical_note: number
        for number, musical_note
        in enumerate(unique_notes)
    }

    int_to_note = {
        number: musical_note
        for number, musical_note
        in enumerate(unique_notes)
    }

    encoded_notes = [
        note_to_int[musical_note]
        for musical_note in notes
    ]

    return (
        encoded_notes,
        note_to_int,
        int_to_note,
        unique_notes
    )


# =============================
# Create Training Sequences
# =============================

def create_sequences(encoded_notes):
    """
    Create input sequences and target notes.
    """

    network_input = []
    network_output = []

    for i in range(
        len(encoded_notes) - SEQUENCE_LENGTH
    ):

        sequence_in = encoded_notes[
            i:i + SEQUENCE_LENGTH
        ]

        sequence_out = encoded_notes[
            i + SEQUENCE_LENGTH
        ]

        network_input.append(
            sequence_in
        )

        network_output.append(
            sequence_out
        )

    return (
        np.array(network_input),
        np.array(network_output)
    )


# =============================
# Normalize Data
# =============================

def prepare_input(
    network_input,
    vocabulary_size
):
    """
    Reshape and normalize input data.
    """

    network_input = network_input.reshape(
        (
            network_input.shape[0],
            network_input.shape[1],
            1
        )
    )

    network_input = (
        network_input /
        float(vocabulary_size)
    )

    return network_input


# =============================
# Save Files
# =============================

def save_data(
    network_input,
    network_output,
    note_to_int,
    int_to_note
):

    np.save(
        os.path.join(
            DATA_FOLDER,
            "network_input.npy"
        ),
        network_input
    )

    np.save(
        os.path.join(
            DATA_FOLDER,
            "network_output.npy"
        ),
        network_output
    )

    with open(
        os.path.join(
            DATA_FOLDER,
            "note_to_int.pkl"
        ),
        "wb"
    ) as file:

        pickle.dump(
            note_to_int,
            file
        )

    with open(
        os.path.join(
            DATA_FOLDER,
            "int_to_note.pkl"
        ),
        "wb"
    ) as file:

        pickle.dump(
            int_to_note,
            file
        )


# =============================
# Main Function
# =============================

def main():

    print("=" * 50)

    print("Reading MIDI Files...")

    notes = extract_notes(
        DATASET_PATH
    )

    print(
        f"Total Notes: {len(notes)}"
    )

    if len(notes) <= SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough notes in the dataset."
        )

    (
        encoded_notes,
        note_to_int,
        int_to_note,
        unique_notes
    ) = encode_notes(notes)

    print(
        f"Unique Notes: {len(unique_notes)}"
    )

    (
        network_input,
        network_output
    ) = create_sequences(
        encoded_notes
    )

    network_input = prepare_input(
        network_input,
        len(unique_notes)
    )

    save_data(
        network_input,
        network_output,
        note_to_int,
        int_to_note
    )

    print("=" * 50)

    print(
        "Preprocessing Finished Successfully!"
    )

    print(
        f"Training Samples: {len(network_input)}"
    )


# =============================
# Run Program
# =============================

if __name__ == "__main__":
    main()