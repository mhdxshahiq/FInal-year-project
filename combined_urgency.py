import math
from Model_Test_Final import get_text_priority
from random_forest_model import get_audio_priority

def get_final_urgency(text_priority, audio_priority):
    # Define priority mapping
    urgency_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
    reverse_mapping = {3: 'High', 2: 'Medium', 1: 'Low'}
    

    # Convert text & audio priorities to numeric values
    text_value = urgency_mapping.get(text_priority, 1)  # Default to 'Low' if not found
    audio_value = urgency_mapping.get(audio_priority, 1)
    
    # Compute final urgency using weighted formula
    final_value = (0.5 * text_value) + (0.5 * audio_value)
    
    # Round to nearest priority level (1, 2, or 3) and map back to label
    final_priority = reverse_mapping[round(final_value)]
    
    return final_priority