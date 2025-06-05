import os
from pathlib import Path
from dotenv import load_dotenv
import slack
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
app_token = os.environ['SLACK_APP_TOKEN']
user_token = os.environ['SLACK_OAUTH_TOKEN']

# SEND MESSAGE TO A CHANNEL
client = slack.WebClient(token=user_token)
def send_message(message):
    client.chat_postMessage(channel='#test', text=message)
def send_message_to_channel(channel, message):
    client.chat_postMessage(channel=channel, text=message)
# send_message_to_channel('#test', 'Hello, world!')

app = App(token=user_token)

@app.event("app_mention")
def handle_app_mentioned_events(event, say):
    say(f"Hey <@{event['user']}>, what's on your mind?")

handler = SocketModeHandler(app, app_token)
handler.start()







