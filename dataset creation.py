import pandas as pd
import numpy as np

# Function to interpret the urgency level
def interpret_urgency(features):
    pitch = features['F0semitoneFrom27.5Hz_sma3nz_amean']
    loudness = features['loudness_sma3_amean']
    jitter = features['jitterLocal_sma3nz_amean']
    hnr = features['HNRdBACF_sma3nz_amean']
    mfcc1 = features['mfcc1_sma3_amean']
    mfcc2 = features['mfcc2_sma3_amean']
    voiced_length = features['MeanVoicedSegmentLengthSec']
    unvoiced_length = features['MeanUnvoicedSegmentLength']

    # High Urgency: High loudness, high pitch, high jitter + fast speech (short voiced & unvoiced segments)
    if (loudness > 1.5 and pitch > 33 and jitter > 0.06) or (voiced_length < 0.23 and unvoiced_length < 0.2) or \
       (mfcc1 > 13 and mfcc2 > 0 and hnr < 5):
        return "High"

    # Low Urgency: Low loudness, low HNR, slow speech (long voiced & unvoiced segments)
    elif (loudness < 0.7 and hnr < 3 and pitch < 30 ) or (voiced_length > 0.2 and unvoiced_length > 0.1) or \
         (mfcc1 < 16 and mfcc2 < -8 ):
        return "Low"

    # Medium Urgency: Default case, moderate values
    else:
        return "Medium"

# Number of samples in the dataset
num_samples = 1000

# Generate synthetic data for each feature with updated ranges
F0semitoneFrom27_5Hz_sma3nz_amean = np.random.uniform(24, 40, size=num_samples)  # Increased pitch range
loudness_sma3_amean = np.random.uniform(0.4, 2, size=num_samples)  # Adjusted loudness range
jitterLocal_sma3nz_amean = np.random.uniform(0.01, 0.09, size=num_samples)  # Jitter range
HNRdBACF_sma3nz_amean = np.random.uniform(1, 9, size=num_samples)  # Lower HNR values
mfcc1_sma3_amean = np.random.uniform(12, 30, size=num_samples)  # MFCC1 range
mfcc2_sma3_amean = np.random.uniform(-15, 9, size=num_samples)  # MFCC2 range
MeanVoicedSegmentLengthSec = np.random.uniform(0.1, 0.6, size=num_samples)  # Speech speed
MeanUnvoicedSegmentLength = np.random.uniform(0.05, 0.3, size=num_samples)  # Pauses in speech

# Create a DataFrame with the features
data = {
    'F0semitoneFrom27.5Hz_sma3nz_amean': F0semitoneFrom27_5Hz_sma3nz_amean,
    'loudness_sma3_amean': loudness_sma3_amean,
    'jitterLocal_sma3nz_amean': jitterLocal_sma3nz_amean,
    'HNRdBACF_sma3nz_amean': HNRdBACF_sma3nz_amean,
    'mfcc1_sma3_amean': mfcc1_sma3_amean,
    'mfcc2_sma3_amean': mfcc2_sma3_amean,
    'MeanVoicedSegmentLengthSec': MeanVoicedSegmentLengthSec,
    'MeanUnvoicedSegmentLength': MeanUnvoicedSegmentLength
}

df = pd.DataFrame(data)

# Apply the interpret_urgency function to classify urgency levels
df['Urgency'] = df.apply(interpret_urgency, axis=1)

# Save to a CSV file
df.to_csv("OpenSmile_Features_Dataset.csv", index=False)

print("Dataset generated and saved as OpenSmile_Features_Dataset.csv")
