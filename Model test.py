import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from transformers import pipeline

# Load the dataset
excel_path = r"C:\Users\Strix\Desktop\MAIN project\final_voicemail_test_samples.xlsx"
df = pd.read_excel(excel_path)

# Define column names (update based on actual column names)
text_column = "Text"  # Column with voicemail transcriptions
label_column = "Urgency"  # Column with urgency labels

# Split dataset into train and test sets (80% train, 20% test)
_, test_df = train_test_split(df, test_size=0.9, stratify=df[label_column], random_state=42)

# Extract test texts and actual labels
test_texts = test_df[text_column].tolist()
actual_labels = test_df[label_column].tolist()

# Load the trained model
classifier = pipeline('text-classification', model=r'C:\Users\Strix\Desktop\MAIN project\results\checkpoint-81', top_k=None)

# Correct label mapping
label_mapping = {
    "LABEL_0": "High",
    "LABEL_1": "Low",
    "LABEL_2": "Medium"
}

# Reverse mapping for numerical labels
label_to_index = {"High": 0, "Medium": 1, "Low": 2}

# Convert actual labels to numerical format for ROC
y_true = [label_to_index[label] for label in actual_labels]

# Get model predictions & probabilities
y_scores = []  # Store probabilities
y_pred = []  # Store predicted labels

for text in test_texts:
    result = classifier(text)
    probs = {label_mapping[item['label']]: item['score'] for item in result[0]}
    y_scores.append([probs["High"], probs["Medium"], probs["Low"]])  # Store probabilities
    y_pred.append(max(probs, key=probs.get))  # Get highest probability label

# Convert lists to NumPy arrays
y_scores = np.array(y_scores)

# Check for mismatched lengths before evaluation
if len(actual_labels) != len(y_pred):
    print(f"Error: Mismatch in actual labels ({len(actual_labels)}) and predicted labels ({len(y_pred)})")
else:
    # Compute classification report
    report = classification_report(actual_labels, y_pred, target_names=["High", "Medium", "Low"])
    print("Classification Report:\n", report)

    # Create Confusion Matrix
    conf_matrix = confusion_matrix(actual_labels, y_pred, labels=["High", "Medium", "Low"])

    # Plot the Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["High", "Medium", "Low"], yticklabels=["High", "Medium", "Low"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # Save as image
    plt.show()

    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    # Compute ROC curve and AUC for each class
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(["High", "Medium", "Low"]):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Text Classification Model")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")  # Save as image
    plt.show()

# Plot urgency score distribution
plt.figure(figsize=(8,5))
sns.boxplot(data=y_scores, showmeans=True)
plt.xticks(ticks=[0,1,2], labels=["High", "Medium", "Low"])
plt.xlabel("Urgency Level")
plt.ylabel("Model Probability Score")
plt.title("Urgency Score Distribution Across Test Samples")
plt.show()
