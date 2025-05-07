import subprocess
import os

def extract_audio_features(audio_file_path, output_file_path, config_path):
    """
    Extract features from an audio file using OpenSMILE and save them to a CSV file.

    Parameters:
    - audio_file_path: str, path to the input audio file (e.g., "input_audio.wav")
    - output_file_path: str, path where the output CSV file should be saved (e.g., "output_features.csv")
    - config_path: str, path to the OpenSMILE config file.
    """
    # Path to the OpenSMILE executable
    opensmile_binary_path = r"C:\Users\Strix\Desktop\MAIN project\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"

    # Check if the audio file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file {audio_file_path} does not exist.")
    
    # Check if OpenSMILE binary exists
    if not os.path.isfile(opensmile_binary_path):
        raise EnvironmentError(f"OpenSMILE binary not found at {opensmile_binary_path}.")

    # Check if the configuration file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    # Build the OpenSMILE command
    command = [
        opensmile_binary_path,
        '-C', config_path,
        '-I', audio_file_path,
        '-O', output_file_path
    ]
    
    # Run the OpenSMILE command
    try:
        subprocess.run(command, check=True)
        print(f"Feature extraction completed. Output saved to {output_file_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Example usage
audio_file = r"C:\Users\Strix\Desktop\MAIN project\Sample Voicemails\WhatsApp Audio 2025-03-18 at 13.20.09_2a8517a8.wav"
output_file = r"C:\Users\Strix\Desktop\MAIN project\OpenSmile Output\OpenSmile Output.csv"
config_file = r"C:\Users\Strix\Desktop\MAIN project\opensmile-3.0.2-windows-x86_64\config\egemaps\v02\eGeMAPSv02.conf"

extract_audio_features(audio_file, output_file, config_file)



import pandas as pd

output_file = r"C:\Users\Strix\Desktop\MAIN project\OpenSmile Output\OpenSmile Output.csv"
features = pd.read_csv(output_file)

print(features.head())
