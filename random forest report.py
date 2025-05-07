import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Load dataset
df = pd.read_csv(r'C:\Users\Strix\Desktop\MAIN project\OpenSmile_test_samples.csv')

# Feature selection
feature_names = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 'jitterLocal_sma3nz_amean', 
                 'HNRdBACF_sma3nz_amean', 'mfcc1_sma3_amean', 'mfcc2_sma3_amean', 
                 'MeanVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength']

X = df[feature_names]  # Features
y = df['Urgency']  # Target variable

# Train-Test Split (Ensuring proper evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# ------------------- Feature Importance Plot -------------------
# ------------------- Feature Importance Plot -------------------
plt.figure(figsize=(6, 4))  # Reduced figure size

importances = model.feature_importances_
sns.barplot(x=importances, y=feature_names, palette='viridis')

plt.xlabel("Feature Importance", fontsize=10)
plt.ylabel("Features", fontsize=10)
plt.title("Random Forest Feature Importance", fontsize=12)

# Adjust font size for ticks
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.tight_layout()  # Ensures everything fits within the figure
plt.show()

# ------------------- Decision Tree Visualization (Matplotlib) -------------------
# Extract one tree from the forest (e.g., the first tree)
tree_classifier = model.estimators_[0]

plt.figure(figsize=(20, 10))
tree.plot_tree(tree_classifier, filled=True, feature_names=feature_names, class_names=model.classes_.astype(str), rounded=True)
plt.title("Decision Tree Visualization (Matplotlib)")
plt.show()

# ------------------- Model Evaluation -------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ------------------- ROC Curve -------------------
# Convert labels to binary format (needed for multi-class problems)
y_bin = label_binarize(y_test, classes=model.classes_)  # Binarizing classes
y_prob = model.predict_proba(X_test)  # Get probability scores

# Plot ROC Curve
plt.figure(figsize=(8, 6))
for i in range(y_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {model.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random classifier)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
