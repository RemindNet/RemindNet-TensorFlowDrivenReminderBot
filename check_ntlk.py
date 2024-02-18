from nltk.classify import extract_datetime
from datetime import datetime
text = "Remind me at 10:30 in the afternoon"
current_datetime = datetime.now()

def handle_datetime(text, current_datetime):
    dates = extract_datetime(text)
    if dates:
        date = dates[0]
        return current_datetime.replace(
            hour=date.hour, minute=date.minute
        )

reminder_datetime = handle_datetime(text, current_datetime)
print(reminder_datetime)
