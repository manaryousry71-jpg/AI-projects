import os
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# 1. Initialize main window
window = tk.Tk()
window.title("Language Translator")
window.geometry("600x680")

languages = {"English": "en","Arabic": "ar","French": "fr", "German": "de","Spanish": "es"}

# 2. Define functions 
def translate():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "Please enter some text to translate.")
        return

    source_code = languages.get(from_language.get(), "auto")
    target_code = languages.get(to_language.get(), "en")

    try:
        translated = GoogleTranslator(source=source_code, target=target_code).translate(text)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, translated)
    except Exception as e:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Error: {e}")

def copy_to_clipboard():
    text = output_text.get("1.0", tk.END).strip()
    if text and not text.startswith("Error:") and not text.startswith("Please enter"):
        window.clipboard_clear()
        window.clipboard_append(text)
        messagebox.showinfo("Success", "Translated text copied to clipboard!")
    else:
        messagebox.showwarning("Warning", "No translated text to copy.")

def text_to_speech():
    text = output_text.get("1.0", tk.END).strip()
    if not text or text.startswith("Error:") or text.startswith("Please enter"):
        messagebox.showwarning("Warning", "No text to speak.")
        return
    
    target_lang = languages.get(to_language.get(), "en")

    try:
        # Generate speech output
        tts = gTTS(text=text, lang=target_lang)
        
        # Save to a temporary MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)

        # Play audio with pygame
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to play audio: {e}")

# 3. Add widgets
title = tk.Label(window, text="Language Translator", font=("Arial", 20, "bold"))
title.pack(pady=10)

# Input Section
text_label = tk.Label(window, text="Enter Text:", font=("Arial", 12))
text_label.pack(pady=(5, 0))

input_text = tk.Text(window, height=5, width=50, font=("Arial", 12))
input_text.pack(pady=5)

# Dropdown Section
lang_options = list(languages.keys())

from_label = tk.Label(window, text="From Language:", font=("Arial", 11))
from_label.pack(pady=(5, 0))
from_language = ttk.Combobox(window, values=lang_options, width=30, state="readonly")
from_language.current(0)
from_language.pack(pady=5)

to_label = tk.Label(window, text="To Language:", font=("Arial", 11))
to_label.pack(pady=(5, 0))
to_language = ttk.Combobox(window, values=lang_options, width=30, state="readonly")
to_language.current(1)
to_language.pack(pady=5)

# Translate Action Button
translate_button = tk.Button(window, text="Translate", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=translate)
translate_button.pack(pady=15)

# Output Section
output_label = tk.Label(window, text="Translated Text:", font=("Arial", 12))
output_label.pack(pady=(5, 0))

output_text = tk.Text(window, height=5, width=50, font=("Arial", 12))
output_text.pack(pady=5)

# Action Frame (Copy & Speak Buttons)
action_frame = tk.Frame(window)
action_frame.pack(pady=10)

copy_button = tk.Button(action_frame, text="📋 Copy", font=("Arial", 11), command=copy_to_clipboard, width=12)
copy_button.grid(row=0, column=0, padx=10)

speak_button = tk.Button(action_frame, text="🔊 Listen", font=("Arial", 11), command=text_to_speech, width=12)
speak_button.grid(row=0, column=1, padx=10)

# 4. Start event loop
window.mainloop()