import opensmile

# Initialize OpenSMILE with the eGeMAPS feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Process a voicemail audio file
features = smile.process_file(r'C:\Users\Strix\Desktop\MAIN project\Sample Voicemails\WhatsApp Audio 2025-03-18 at 15.40.12_8d958239.wav')

# Save the features to a CSV file
output_file_path = r"C:\Users\Strix\Desktop\MAIN project\OpenSmile Output\OpenSmile Output using library.csv"
features.to_csv(output_file_path)

print('....................................')



#############################################################