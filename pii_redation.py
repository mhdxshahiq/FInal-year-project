import spacy
import re
from pydub import AudioSegment
from pydub import AudioSegment
import os

# Load Spacy's NER model
nlp = spacy.load("en_core_web_sm")


def hide_pii(text):
    # Mask emails (handles spaces before/after @ and .)
    text = re.sub(
        r"(?i)\b[a-zA-Z]+\s+[a-zA-Z0-9._%+-]*\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}\b",
        "[EMAIL_HIDDEN]",
        text
    )


    # Mask phone numbers (various formats)
    text = re.sub(
        r"(?i)\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,6}\b",
        "[PHONE_HIDDEN]",
        text
    )

    # Mask Patient ID (Example: Patient ID: 123456789)
    text = re.sub(r"(?i)Patient ID[:\s]+[\w-]+", "[ID_HIDDEN]", text)

    # Mask Medical Record Number (MRN: 987-654-321)
    text = re.sub(r"(?i)Medical ID[:\s]+[\w-]+", "[MRN_HIDDEN]", text)

    # Mask Insurance Number (Example: Insurance: ABCD123456)
    text = re.sub(r"(?i)Insurance[:\s]+[A-Z0-9-]+", "[INSURANCE_HIDDEN]", text)

    # Mask Social Security Numbers (Example: SSN: 123-45-6789)
    text = re.sub(r"(?i)SSN[:\s]+\d{3}-\d{2}-\d{4}", "[SSN_HIDDEN]", text)

    # Use spaCy for Named Entity Recognition (NER)
    doc = nlp(text)
    for entity in doc.ents:
        if entity.label_ in {"GPE", "LOC"}:
            text = text.replace(entity.text, f"[{entity.label_}_HIDDEN]")

    return text


def replace_pii_with_beep(original_audio_path, text):
    """Replace PII in text with a beep sound in the audio and save it in static/pii_voicemails."""

    # Ensure the original file exists
    if not os.path.exists(original_audio_path):
        print(f"Error: File not found - {original_audio_path}")
        return None  # Return None if the file doesn't exist

    # Ensure the output directory exists
    output_dir = "static/pii_voicemails"
    os.makedirs(output_dir, exist_ok=True)

    # Load original WAV audio
    original_audio = AudioSegment.from_wav(original_audio_path)

    # Load a beep sound (trim to 700ms for better coverage)
    beep = AudioSegment.from_wav("C:/Users/Strix/Desktop/MAIN project/beep-125033.wav")[:700]

    # Get anonymized text
    anonymized_text = hide_pii(text)

    # Split original and anonymized text into words
    original_words = text.split()
    anonymized_words = anonymized_text.split()

    # Estimate word duration
    total_duration = len(original_audio)  # Audio duration in milliseconds
    word_duration = total_duration / max(len(original_words), 1)  # Avoid division by zero

    # Find PII word positions (covering the previous word if necessary)
    pii_positions = []
    last_pii_index = -1

    for i, (original_word, anonymized_word) in enumerate(zip(original_words, anonymized_words)):
        if anonymized_word in ["[EMAIL_HIDDEN]", "[PHONE_HIDDEN]", "[LOCATION_HIDDEN]","[ID_HIDDEN]","[GPE_HIDDEN]","[LOC_HIDDEN]","[MRN_HIDDEN]"]:
            start_time = int(i * word_duration)  # Approximate start time in ms

            # Cover previous word too (e.g., "Muhammad 123@gmail.com" → "Muhammad [EMAIL_HIDDEN]")
            if last_pii_index != i - 1 and i > 0:
                start_time = int((i - 1) * word_duration)

            pii_positions.append(start_time)
            last_pii_index = i  # Update last PII index

    # Create a mutable copy of the original audio
    edited_audio = original_audio

    # Mute original voice and add beep at each PII position
    for start_time in pii_positions:
        end_time = start_time + len(beep)  # Beep duration
        silence = AudioSegment.silent(duration=len(beep))  # Create silence of beep duration

        # Replace the segment with silence
        edited_audio = edited_audio[:start_time] + silence + edited_audio[end_time:]

        # Overlay beep on the muted section
        edited_audio = edited_audio.overlay(beep, position=start_time)

    # Generate anonymized file path
    anonymized_audio_filename = os.path.basename(original_audio_path).replace(".wav", "_pii.wav")
    anonymized_audio_path = os.path.join(output_dir, anonymized_audio_filename)

    # Save the anonymized audio
    edited_audio.export(anonymized_audio_path, format="wav")

    # Return the relative path for use in the frontend
    return f"pii_voicemails/{anonymized_audio_filename}"



#original_audio_path = r'C:\Users\Strix\Desktop\MAIN project\static\voicemails\voicemail_RE786e858c0b1b7f7d180f5de028f7c24c.wav'
#text = 'hello I need a general check up please contact me in this email id Muhammad 123@gmail.com thank you'
#a = replace_pii_with_beep(original_audio_path, text)
#print(a)

# Sample input text
#text = "hello I need a general check up please contact me in this email id Muhammad 123@gmail.com thank you"
#anonymized_text = hide_pii(text)
#print(f"Anonymized Text: {anonymized_text}")
