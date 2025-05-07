import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download required NLTK data
nltk.download('stopwords')

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

class BertClassifier(nn.Module):
    def __init__(self, n_classes, bert_model='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
        # Move model to GPU if available
        self.to(device)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)

class HospitalVoicemailPrioritizer:
    def __init__(self):
        self.device = device
        print(f"Initializing model on: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.urgency_encoder = LabelEncoder()
        self.department_encoder = LabelEncoder()
        self.stop_words = set(stopwords.words('english')) - {'urgent', 'immediately', 'emergency'}
        
        self.urgency_classifier = None
        self.department_classifier = None
    
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def train_model(self, model, train_loader, val_loader, epochs=5):
        print(f"Training on {self.device}")
        model = model.to(self.device)
        
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
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
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['label']
                    
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    _, preds = torch.max(outputs, dim=1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_accuracy = (np.array(val_preds) == np.array(val_labels)).mean()
            print(f'Validation Accuracy: {val_accuracy:.4f}')
    
    def fit(self, df, batch_size=16):
        """Train the urgency and department classifiers."""
        texts = [self.preprocess_text(text) for text in df['Text']]
        
        # Encode labels
        urgency_labels = self.urgency_encoder.fit_transform(df['Urgency'])
        department_labels = self.department_encoder.fit_transform(df['Department'])
        
        # Create datasets with stratification
        train_texts, val_texts, train_urgency, val_urgency = train_test_split(
            texts, urgency_labels, test_size=0.2, random_state=42, stratify=urgency_labels
        )
        
        # Create data loaders with pin_memory for faster GPU transfer
        train_urgency_dataset = VoicemailDataset(train_texts, train_urgency, self.tokenizer)
        val_urgency_dataset = VoicemailDataset(val_texts, val_urgency, self.tokenizer)
        
        train_urgency_loader = DataLoader(
            train_urgency_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True
        )
        val_urgency_loader = DataLoader(
            val_urgency_dataset, 
            batch_size=batch_size,
            pin_memory=True
        )
        
        # Initialize and train urgency classifier
        print("\nTraining Urgency Classifier:")
        self.urgency_classifier = BertClassifier(len(self.urgency_encoder.classes_))
        self.train_model(
            self.urgency_classifier,
            train_urgency_loader,
            val_urgency_loader,
            epochs=5
        )
        
        # Train department classifier
        train_texts, val_texts, train_dept, val_dept = train_test_split(
            texts, department_labels, test_size=0.2, random_state=42, stratify=department_labels
        )
        
        train_dept_dataset = VoicemailDataset(train_texts, train_dept, self.tokenizer)
        val_dept_dataset = VoicemailDataset(val_texts, val_dept, self.tokenizer)
        
        train_dept_loader = DataLoader(
            train_dept_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True
        )
        val_dept_loader = DataLoader(
            val_dept_dataset, 
            batch_size=batch_size,
            pin_memory=True
        )
        
        print("\nTraining Department Classifier:")
        self.department_classifier = BertClassifier(len(self.department_encoder.classes_))
        self.train_model(
            self.department_classifier,
            train_dept_loader,
            val_dept_loader,
            epochs=5
        )
    
    def predict(self, texts, batch_size=16):
        """Predict urgency levels and departments for new voicemails."""
        self.urgency_classifier.eval()
        self.department_classifier.eval()
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        dataset = VoicemailDataset(
            processed_texts,
            [0] * len(processed_texts),
            self.tokenizer
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            pin_memory=True
        )
        
        urgency_preds = []
        department_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                # Urgency predictions
                outputs = self.urgency_classifier(input_ids, attention_mask, token_type_ids)
                _, preds = torch.max(outputs, dim=1)
                urgency_preds.extend(preds.cpu().numpy())
                
                # Department predictions
                outputs = self.department_classifier(input_ids, attention_mask, token_type_ids)
                _, preds = torch.max(outputs, dim=1)
                department_preds.extend(preds.cpu().numpy())
        
        urgency_labels = self.urgency_encoder.inverse_transform(urgency_preds)
        department_labels = self.department_encoder.inverse_transform(department_preds)
        
        return urgency_labels, department_labels

def evaluate_system(data_path):
    """Evaluate the hospital voicemail prioritization system."""
    df = pd.read_csv(data_path)
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Urgency']
    )
    
    prioritizer = HospitalVoicemailPrioritizer()
    prioritizer.fit(train_df)
    
    pred_urgency, pred_department = prioritizer.predict(test_df['Text'])
    
    print("\nUrgency Classification Report:")
    print(classification_report(test_df['Urgency'], pred_urgency))
    print("\nDepartment Classification Report:")
    print(classification_report(test_df['Department'], pred_department))
    
    return prioritizer, test_df

if __name__ == "__main__":
    data_path = r"C:\Users\Strix\Desktop\MAIN project\Hospital_Voicemail_Transcription.csv"
    prioritizer, test_df = evaluate_system(data_path)
    
    new_voicemails = [
        "Patient reporting severe chest pain in emergency room, need immediate assistance",
        "Calling to schedule a routine checkup next week",
        "The pharmacy is running low on critical medications, need urgent resupply"
    ]
    
    pred_urgency, pred_department = prioritizer.predict(new_voicemails)
    
    results = pd.DataFrame({
        'Text': new_voicemails,
        'Predicted_Urgency': pred_urgency,
        'Predicted_Department': pred_department
    })
    print("\nPredictions for new voicemails:")
    print(results)