import requests

ACCOUNT_SID = #AUTH SID
AUTH_TOKEN = #auth token

# Get all recordings
url = f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Recordings.json"
response = requests.get(url, auth=(ACCOUNT_SID, AUTH_TOKEN))

if response.status_code == 200:
    recordings = response.json()["recordings"]
    for rec in recordings:
        rec_sid = rec["sid"]
        delete_url = f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT_SID}/Recordings/{rec_sid}.json"
        del_response = requests.delete(delete_url, auth=(ACCOUNT_SID, AUTH_TOKEN))
        
        if del_response.status_code == 204:
            print(f"Deleted recording: {rec_sid}")
        else:
            print(f"Failed to delete recording {rec_sid}: {del_response.status_code}")
else:
    print("Failed to fetch recordings:", response.text)