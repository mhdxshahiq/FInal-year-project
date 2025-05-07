import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load Excel file
excel_path = r"C:\Users\Strix\Desktop\MAIN project\Latest_Hospital_Voicemail_Transcription.xlsx"
df = pd.read_excel(excel_path)

# Drop 'Department' column as it's not needed for this task
df = df.drop(columns=['Department'])

# Encode 'Urgency' column into numeric labels
df['Urgency'] = df['Urgency'].astype('category').cat.codes

print(df)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split dataset into train and test
dataset = dataset.train_test_split(test_size=0.2, seed=42)  # Set random seed for reproducibility

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples['Text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Rename label column for Trainer compatibility
tokenized_dataset = tokenized_dataset.rename_column("Urgency", "labels")

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Pass the compute_metrics function
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)  # Print the evaluation metrics

