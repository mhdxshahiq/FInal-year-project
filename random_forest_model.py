import joblib
import subprocess
import os
import opensmile
import pandas as pd


# Load the trained model
clf = joblib.load('random_forest_model.pkl')

# Paths
OPENSMILE_BINARY_PATH = r"C:\Users\Strix\Desktop\MAIN project\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"
CONFIG_PATH = r"C:\Users\Strix\Desktop\MAIN project\opensmile-3.0.2-windows-x86_64\config\egemaps\v02\eGeMAPSv02.conf"
CSV_OUTPUT_PATH = r"C:\Users\Strix\Desktop\MAIN project\OpenSmile Output\OpenSmile Output.csv"



# Initialize OpenSMILE with the eGeMAPS feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
csv_output_path = r"C:\Users\Strix\Desktop\MAIN project\OpenSmile Output\OpenSmile Output using library.csv"



def extract_audio_featuresS(audio_file_path):
    """
    Extracts features from an audio file using OpenSMILE and saves them to the specified CSV output path.
    The function returns the extracted features as a Pandas DataFrame.

    Parameters:
    - audio_file_path (str): Path to the input audio file.

    Returns:
    - pandas.DataFrame: Extracted feature values (last row).
    """

    # Check if the audio file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file {audio_file_path} does not exist.")
    
    # Check if OpenSMILE binary exists
    if not os.path.isfile(OPENSMILE_BINARY_PATH):
        raise EnvironmentError(f"OpenSMILE binary not found at {OPENSMILE_BINARY_PATH}.")

    # Check if the configuration file exists
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file {CONFIG_PATH} does not exist.")

    # Build the OpenSMILE command
    command = [
        OPENSMILE_BINARY_PATH,
        '-C', CONFIG_PATH,
        '-I', audio_file_path,
        '-csvoutput', CSV_OUTPUT_PATH  # ✅ Correct flag for OpenSMILE
    ]

    # Run OpenSMILE feature extraction
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenSMILE error: {e.stderr.decode()}")

    # Load extracted features into a DataFrame
    try:
        features_df = pd.read_csv(CSV_OUTPUT_PATH)

        # Ensure that the first row isn't an extra metadata row from OpenSMILE
        if features_df.shape[0] > 1 and "name" in features_df.columns[0].lower():
            features_df = pd.read_csv(CSV_OUTPUT_PATH, skiprows=[0])  # Skip metadata row

        # Keep only the last row
        last_row = features_df.iloc[[-1]]  # Select last row as DataFrame

        # Remove the extracted row from the file (to avoid duplication on next run)
        features_df = features_df.iloc[:-1]  # Remove last row
        features_df.to_csv(CSV_OUTPUT_PATH, index=False)  # Save updated file

    except Exception as e:
        raise RuntimeError(f"Failed to load extracted features: {e}")

    return last_row  # ✅ Return only the last row


# Initialize OpenSMILE with the eGeMAPS feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)





def extract_audio_featuresL(audio_file_path):
    """
    Extracts features from an audio file using OpenSMILE and saves them to a CSV file.
    Returns the last row of extracted features as a DataFrame.

    Parameters:
    - audio_file_path (str): Path to the voicemail audio file.
    - output_csv_path (str): Path where the extracted features will be saved.

    Returns:
    - pandas.DataFrame: Extracted feature values (last row).
    """
    # Process the audio file with OpenSMILE
    features = smile.process_file(audio_file_path)

    # Save the extracted features to CSV
    features.to_csv(csv_output_path)

    # Keep only the last row
    last_row = features.iloc[[-1]]  # Ensures a DataFrame, not a Series

    return last_row





def get_audio_priority(audio_file_path):
    """
    Extracts audio features using OpenSMILE and predicts urgency using a trained model.

    Parameters:
    - audio_file_path (str): Path to the voicemail audio file.

    Returns:
    - pandas.Series: Predicted urgency values.
    """

    # Extract features (last row only)
    new_voicemails_df = extract_audio_featuresL(audio_file_path)

    # Required features for the model
    required_features = [
        'F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 
        'jitterLocal_sma3nz_amean', 'HNRdBACF_sma3nz_amean', 
        'mfcc1_sma3_amean', 'mfcc2_sma3_amean', 
        'MeanVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength'
    ]


    # Select only the required features
    new_voicemail_features = new_voicemails_df[required_features]

    # Predict urgency
    predicted_urgency = clf.predict(new_voicemail_features)

    return str(predicted_urgency[0])


#path =r"C:\Users\Strix\Desktop\MAIN project\Sample Voicemails\WhatsApp Audio 2025-03-18 at 13.02.01_2911823b.wav"
#a = extract_audio_featuresL(path)
#b = get_audio_priority(path)
#print(a)
#print(b)