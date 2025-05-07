import speech_recognition as sr
from transformers import pipeline

# Define the correct label mapping
label_mapping = {
    "LABEL_0": "High",
    "LABEL_1": "Low",
    "LABEL_2": "Medium"
}

# Load the trained model (use the correct checkpoint path)
classifier = pipeline('text-classification', model=r'C:\Users\Strix\Desktop\MAIN project\results\checkpoint-81')

# Function to capture audio input and convert to text
import speech_recognition as sr

def get_audio_transcription(audio_file_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file_path) as source:
        try:
            audio = recognizer.record(source)  # Read the entire file
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "aaa"  # Could not understand audio
        except sr.RequestError:
            return "aaa"  # Issue with the recognition service

# Function to classify urgency of text
def get_text_priority(text):
    # Classify the text
    result = classifier(text)

    # Map the label to string
    for r in result:
        r['label'] = label_mapping[r['label']]

    return result[0]['label'] if result else 'Low' 

def get_text_urgency(text):
    # Classify the text
    result = classifier(text)

    # Map the label to string
    for r in result:
        r['label'] = label_mapping[r['label']]

    return result[0]['score'] if result else 0.1


#a = get_audio_transcription(r'C:\Users\Strix\Desktop\MAIN project\Sample Voicemails\WhatsApp Audio 2025-03-18 at 13.13.23_92833000.wav')
#print(a)