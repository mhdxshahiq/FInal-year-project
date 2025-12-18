import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download required NLTK data
nltk.download('stopwords')

# Dataset Class
class VoicemailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# BERT Classifier Model
class BertClassifier(nn.Module):
    def __init__(self, n_classes, bert_model='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)

# Training Function
def train_model(model, train_loader, val_loader, epochs=15, save_path="department_model.pt"):
    print(f"\nTraining on {device}")
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f'Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Average training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label']
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                _, preds = torch.max(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_accuracy = (np.array(val_preds) == np.array(val_labels)).mean()
        print(f'Validation Accuracy: {val_accuracy:.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Main Function
def train_department_classifier(data_path, batch_size=16):
    """Train the department classifier model."""
    df = pd.read_excel(data_path)

    # Initialize tokenizer & label encoder
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    department_encoder = LabelEncoder()
    
    # Preprocess text
    stop_words = set(stopwords.words('english')) - {'urgent', 'immediately', 'emergency'}
    def preprocess_text(text):
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    
    texts = [preprocess_text(text) for text in df['Text']]
    department_labels = department_encoder.fit_transform(df['Department'])
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, department_labels, test_size=0.2, random_state=42, stratify=department_labels
    )
    
    # Create datasets
    train_dataset = VoicemailDataset(train_texts, train_labels, tokenizer)
    val_dataset = VoicemailDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize and train the department classifier
    print("\nTraining Department Classifier:")
    department_classifier = BertClassifier(len(department_encoder.classes_))
    train_model(department_classifier, train_loader, val_loader, epochs=15)
    
    return department_classifier, department_encoder

if __name__ == "__main__":
    data_path = r"C:\Users\Strix\Desktop\MAIN project\Latest_Hospital_Voicemail_Transcription.xlsx"
    department_classifier, department_encoder = train_department_classifier(data_path)

    # Example Predictions
    test_texts = [
        "Patient experiencing severe allergic reaction, requires urgent care",
        "Routine inquiry about health insurance policy",
        "Medical staff reporting shortage of syringes in the pharmacy department"
    ]
    
    # Process test texts
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = VoicemailDataset(test_texts, [0] * len(test_texts), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1)

    department_classifier.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            outputs = department_classifier(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    predicted_departments = department_encoder.inverse_transform(predictions)
    
    # Display results
    results = pd.DataFrame({
        'Text': test_texts,
        'Predicted_Department': predicted_departments
    })
    print("\nPredictions for test voicemails:")
    print(results)
