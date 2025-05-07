import os
import csv
import requests
from flask import Flask, jsonify, render_template,redirect
from twilio.rest import Client
from flask import Flask, Response,request
from twilio.twiml.voice_response import VoiceResponse

from Model_Test_Final import get_audio_transcription, get_text_urgency, get_text_priority
from random_forest_model import get_audio_priority
from combined_urgency import get_final_urgency
from dept_model_test import get_department
from pii_redation import hide_pii,replace_pii_with_beep

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return redirect('/download-voicemails')

# Twilio credentials
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)

# Path to CSV file
csv_file_path = r'C:\Users\strix\Desktop\MAIN project\voicemails.csv'

# Ensure the CSV file exists and write headers only once
def initialize_csv():
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as f:
            fieldnames = ["Recording SID", "Caller Number", "Caller Location", "Date and Time", "Duration (Seconds)", "Transcription", "Priority Label", "Text Urgency", "Department", "Audio Urgency", "Final Priority","Audio Path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

# Function to download voicemail recordings
def download_recording(recording_url, filename):
    """Downloads a voicemail recording from Twilio with authentication."""
    response = requests.get(recording_url, auth=(account_sid, auth_token))
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
        return filename
    else:
        print(f"Failed to download recording. Status code: {response.status_code}, Reason: {response.reason}")
        return None

# Function to get caller location using Twilio Lookup API
def get_caller_location(phone_number):
    """Uses Twilio Lookup API to get the caller's location."""
    try:
        lookup = client.lookups.v2.phone_numbers(phone_number).fetch()
        return lookup.country_code  # Returns the country code (e.g., US, IN, etc.)
    except Exception as e:
        print(f"Error fetching caller location: {e}")
        return "Unknown Location"

# Function to write voicemail data to CSV
def write_voicemail_to_csv(recording_sid, caller_number, caller_location, date_time, duration, transcription, text_priority, text_urgency, department, audio_urgency, final_priority, audio_path):
    try:
        with open(csv_file_path, mode='a', newline='') as file:
            fieldnames = ["Recording SID", "Caller Number", "Caller Location", "Date and Time", "Duration (Seconds)","Transcription","Priority Label","Text Urgency","Department","Audio Urgency","Final Priority","Audio Path"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({
                "Recording SID": recording_sid,
                "Caller Number": caller_number,
                "Caller Location": caller_location,
                "Date and Time": date_time,
                "Duration (Seconds)": duration, 
                "Transcription": transcription,
                "Priority Label": text_priority,
                "Text Urgency": text_urgency,
                "Department": department,
                "Audio Urgency": audio_urgency,
                "Final Priority": final_priority,
                "Audio Path": audio_path
            })
    except Exception as e:
        print(f"Error writing voicemail data to CSV: {e}")


def voicemail_exists(recording_sid):
    """Checks if a voicemail SID already exists in the CSV file."""
    if not os.path.exists(csv_file_path):
        return False  # CSV doesn't exist, so no duplicates

    with open(csv_file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return any(row["Recording SID"] == recording_sid for row in reader)
    

# Route to download all voicemail recordings and prioritize them
@app.route("/download-voicemails", methods=["GET"])
def download_voicemails():
    """Fetches and downloads all voicemail recordings from Twilio, prioritizes them, and stores them in a CSV file."""
    try:
        recordings = client.recordings.list()

        if not recordings:
            return jsonify({"message": "No recordings found"})

        # Ensure the voicemails folder exists
        if not os.path.exists("static/voicemails"):
            os.makedirs("static/voicemails")

        # Initialize CSV file with headers if not done already
        initialize_csv()

        # Iterate over recordings and download each one
        for recording in recordings:
            if voicemail_exists(recording.sid):
                continue

            recording_url = f"https://api.twilio.com{recording.uri.replace('.json', '.wav')}"
            file_path = os.path.join('static', 'voicemails', f"voicemail_{recording.sid}.wav")

            filename = download_recording(recording_url, file_path)
            
            if filename:
                # Retrieve call details using Call SID
                call = client.calls(recording.call_sid).fetch()
                recording_sid = recording.sid
                caller_number = call.from_formatted

                # Get caller location
                caller_location = get_caller_location(caller_number)

                # Dummy transcription and urgency scores
                transcription = get_audio_transcription(file_path)
                text_urgency_score = get_text_urgency(transcription)
                text_priority = get_text_priority(transcription)
                department = get_department(transcription)
                audio_urgency_score = get_audio_priority(file_path)
                final_priority = get_final_urgency(text_priority,audio_urgency_score)

                # Write voicemail data to CSV
                write_voicemail_to_csv(
                    recording_sid,
                    caller_number,
                    caller_location,
                    str(recording.date_created),
                    recording.duration,
                    transcription,
                    text_priority,
                    text_urgency_score,
                    department,
                    audio_urgency_score,
                    final_priority,
                    f"voicemails/voicemail_{recording.sid}.wav"
                )

        return jsonify({"message": "Voicemails downloaded and saved to CSV", "total_recordings": len(recordings)})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/delete_voicemail', methods=['POST'])
def delete_voicemail():
    data = request.json
    recording_sid = data.get("recording_sid")  # Retrieve the Recording SID from the request

    if not recording_sid:
        return jsonify({"error": "No Recording SID provided"}), 400
    
    try:
        client.recordings(recording_sid).delete()
        print(f"✅ Deleted Twilio voicemail: {recording_sid}")
    except Exception as e:
        print(f"❌ Failed to delete Twilio voicemail: {e}")

    # Path to voicemail audio file
    audio_path = os.path.join("static", "voicemails", f"voicemail_{recording_sid}.wav")

    # Read the CSV and filter out the voicemail to delete
    updated_rows = []
    voicemail_found = False

    with open(csv_file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames  # Store the headers
        for row in reader:
            if row["Recording SID"] == recording_sid:
                voicemail_found = True
            else:
                updated_rows.append(row)  # Keep all other rows

    if not voicemail_found:
        return jsonify({"error": "Voicemail not found"}), 404

    # Write the updated data back to the CSV file
    with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    # Delete the voicemail audio file if it exists
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return jsonify({"message": "Voicemail deleted successfully"}), 200


@app.route('/forward_voicemail', methods=['POST'])
def forward_voicemail():
    data = request.json
    recording_sid = data.get("recording_sid")
    department = data.get("department")

    if not recording_sid or not department:
        return jsonify({"error": "Missing Recording SID or Department"}), 400

    # Read the CSV and update the department field
    updated_rows = []
    voicemail_found = False

    with open(csv_file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["Recording SID"] == recording_sid:
                row["Department"] = department  # Update department
                voicemail_found = True
            updated_rows.append(row)

    if not voicemail_found:
        return jsonify({"error": "Voicemail not found"}), 404

    # Write the updated data back to CSV
    with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return jsonify({"message": f"Voicemail forwarded to {department}"}), 200


# Route to view prioritized voicemails
@app.route("/index", methods=["GET"])
def view_voicemails():
    """Displays prioritized voicemails in a web interface."""
    voicemails = []
    try:
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as f:
                reader = csv.DictReader(f)  # Use DictReader to read CSV as dictionaries
                for row in reader:
                    voicemails.append({
                        "recording_sid": row["Recording SID"],
                        "caller_number": row["Caller Number"],
                        "caller_location": row["Caller Location"],
                        "date_time": row["Date and Time"],
                        "duration": row["Duration (Seconds)"],
                        "transcription":row["Transcription"],
                        "text_urgency": f"{float(row['Text Urgency']) * 100:.2f}%",
                        "department":  row["Department"],
                        "final_priority": row["Final Priority"],
                        "audio_path": row["Audio Path"],
                    })

        return render_template("index.html", voicemails=voicemails)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/pharmacy-dashboard")
def pharmacy_dashboard():
    voicemails = []
    try:
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:           
                    # Apply PII redaction to audio
                    original_audio_path = os.path.join("static", row["Audio Path"])
                    transcription = row["Transcription"]
                
                    # Apply PII redaction to audio
                    pii_audio_path = replace_pii_with_beep(original_audio_path, transcription)
                    voicemails.append({
                        "recording_sid": row["Recording SID"],
                        "caller_number": row["Caller Number"],
                        "caller_location": row["Caller Location"] if row["Caller Location"].strip() else "INDIA",
                        "date_time": row["Date and Time"],
                        "duration": row["Duration (Seconds)"],
                        "transcription":hide_pii(row["Transcription"]),
                        "text_urgency": float(row["Text Urgency"]),  # Convert to float for sorting
                        "text_urgency_display": f"{float(row['Text Urgency']) * 100:.2f}%",  # Keep string format for UI
                        "department":  row["Department"],
                        "final_priority": row["Final Priority"],
                        "audio_path": pii_audio_path if pii_audio_path else row["Audio Path"],  # Use redacted audio if available,
                    })
        # Filter voicemails to only include Pharmacy department
        pharmacy_voicemails = [v for v in voicemails if v["department"].lower() == "pharmacy"]

        # Sort by Priority (High → Medium → Low), then by Urgency Score (Descending)
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        pharmacy_voicemails = sorted(
            pharmacy_voicemails, 
            key=lambda x: (priority_order[x["final_priority"]], -x["text_urgency"])
        )

        return render_template("pharmacy-dashboard.html", voicemails=pharmacy_voicemails)
    
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/doctors-dashboard")
def doctors_dashboard():
    voicemails = []
    try:
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Apply PII redaction to audio
                    original_audio_path = os.path.join("static", row["Audio Path"])
                    transcription = row["Transcription"]
                
                    # Apply PII redaction to audio
                    pii_audio_path = replace_pii_with_beep(original_audio_path, transcription)
                    voicemails.append({
                        "recording_sid": row["Recording SID"],
                        "caller_number": row["Caller Number"],
                        "caller_location": row["Caller Location"] if row["Caller Location"].strip() else "INDIA",
                        "date_time": row["Date and Time"],
                        "duration": row["Duration (Seconds)"],
                        "transcription":hide_pii(row["Transcription"]),
                        "text_urgency": float(row["Text Urgency"]),  # Convert to float for sorting
                        "text_urgency_display": f"{float(row['Text Urgency']) * 100:.2f}%",  # Keep string format for UI
                        "department":  row["Department"],
                        "final_priority": row["Final Priority"],
                        "audio_path": pii_audio_path if pii_audio_path else row["Audio Path"],  # Use redacted audio if available,
                    })
        # Filter voicemails to only include Pharmacy department
        pharmacy_voicemails = [v for v in voicemails if v["department"].lower() == "doctors"]

        # Sort by Priority (High → Medium → Low), then by Urgency Score (Descending)
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        pharmacy_voicemails = sorted(
            pharmacy_voicemails, 
            key=lambda x: (priority_order[x["final_priority"]], -x["text_urgency"])
        )

        return render_template("doctors-dashboard.html", voicemails=pharmacy_voicemails)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route("/emergency-dashboard")
def emergency_dashboard():
    voicemails = []
    try:
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                                        # Apply PII redaction to audio
                    original_audio_path = os.path.join("static", row["Audio Path"])
                    transcription = row["Transcription"]
                
                    # Apply PII redaction to audio
                    pii_audio_path = replace_pii_with_beep(original_audio_path, transcription)

                    voicemails.append({
                        "recording_sid": row["Recording SID"],
                        "caller_number": row["Caller Number"],
                        "caller_location": row["Caller Location"] if row["Caller Location"].strip() else "INDIA",
                        "date_time": row["Date and Time"],
                        "duration": row["Duration (Seconds)"],
                        "transcription":hide_pii(row["Transcription"]),
                        "text_urgency": float(row["Text Urgency"]),  # Convert to float for sorting
                        "text_urgency_display": f"{float(row['Text Urgency']) * 100:.2f}%",  # Keep string format for UI
                        "department":  row["Department"],
                        "final_priority": row["Final Priority"],
                        "audio_path": pii_audio_path if pii_audio_path else row["Audio Path"],  # Use redacted audio if available,
                    })
        # Filter voicemails to only include Pharmacy department
        pharmacy_voicemails = [v for v in voicemails if v["department"].lower() == "emergency"]

        # Sort by Priority (High → Medium → Low), then by Urgency Score (Descending)
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        pharmacy_voicemails = sorted(
            pharmacy_voicemails, 
            key=lambda x: (priority_order[x["final_priority"]], -x["text_urgency"])
        )

        return render_template("emergency-dashboard.html", voicemails=pharmacy_voicemails)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route("/reception-dashboard")
def reception_dashboard():
    voicemails = []
    try:
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    voicemails.append({
                        "recording_sid": row["Recording SID"],
                        "caller_number": row["Caller Number"],
                        "caller_location": row["Caller Location"] if row["Caller Location"].strip() else "INDIA",
                        "date_time": row["Date and Time"],
                        "duration": row["Duration (Seconds)"],
                        "transcription": row["Transcription"],
                        "text_urgency": float(row["Text Urgency"]),  # Convert to float for sorting
                        "text_urgency_display": f"{float(row['Text Urgency']) * 100:.2f}%",  # Keep string format for UI
                        "department": row["Department"],
                        "final_priority": row["Final Priority"],
                        "audio_path": row["Audio Path"],
                    })

        # Sorting: High → Medium → Low, then by Text Urgency Score (Descending)
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        voicemails = sorted(
            voicemails, 
            key=lambda x: (priority_order[x["final_priority"]], -x["text_urgency"])  # Sort by priority, then urgency score
        )

        return render_template("reception-dashboard.html", voicemails=voicemails)
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
