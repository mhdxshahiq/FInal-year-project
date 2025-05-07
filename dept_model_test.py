import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model class used for training
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

# Load the trained model
num_classes = 4  # Update this based on your dataset
model = BertClassifier(num_classes)
model.load_state_dict(torch.load("department_model.pt", map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define department labels (adjust as per your training data)
department_labels = ["Doctors","Emergency","Pharmacy","Reception"]

def get_department(transcription):
    """Predicts the department for a given voicemail transcription."""
    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        transcription,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move input to the device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_ids, attention_mask, token_type_ids)
        predicted_class = torch.argmax(output, dim=1).cpu().item()
    
    return department_labels[predicted_class]

# Example usage
#transcription ="Um, my wife’s having severe stomach pain, like, we’re heading to emergency."
#predicted_department = get_department(transcription)
#print(f"Predicted Department: {predicted_department}")
