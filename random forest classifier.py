import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the data from CSV
# Assuming your CSV file is named 'voicemail_data.csv' and it includes the 9 features and the 'urgency' label column
df = pd.read_csv(r'C:\Users\Strix\Desktop\MAIN project\OpenSmile_Features_Dataset.csv')

# Features and labels
X = df[['F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 'jitterLocal_sma3nz_amean', 
        'HNRdBACF_sma3nz_amean', 'mfcc1_sma3_amean', 'mfcc2_sma3_amean', 
        'MeanVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength']]
y = df['Urgency']  # This is the label column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate and print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained classifier to a file
joblib.dump(clf, 'random_forest_model.pkl')
print("Model saved successfully as random_forest_model.pkl")


########################## # # # # #prediction# # # # # # ###################################

import pandas as pd

# Load the CSV file with multiple rows (new voicemails)
new_voicemails_df = pd.read_csv(r'c:\Users\Strix\Desktop\MAIN project\OpenSmile Output\OpenSmile Output.csv')

# Ensure the dataframe contains only the 8 features that were used during training
new_voicemail_features = new_voicemails_df[['F0semitoneFrom27.5Hz_sma3nz_amean', 
                                            'loudness_sma3_amean', 
                                            'jitterLocal_sma3nz_amean', 
                                            'HNRdBACF_sma3nz_amean', 
                                            'mfcc1_sma3_amean', 
                                            'mfcc2_sma3_amean', 
                                            'MeanVoicedSegmentLengthSec', 
                                            'MeanUnvoicedSegmentLength']]


# Predict the urgency/emotion for each row in the dataframe
predicted_urgency = clf.predict(new_voicemail_features)

# Output predictions
new_voicemails_df['Predicted Urgency'] = predicted_urgency

# Optionally, save the predictions to a new CSV file
new_voicemails_df.to_csv('predicted_voicemails.csv', index=False)

# Print the dataframe with predicted urgency labels
print(new_voicemails_df[['Predicted Urgency']])

