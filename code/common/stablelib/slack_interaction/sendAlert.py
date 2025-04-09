
from slack_sdk.webhook import WebhookClient
from common.config import SLACK_ALERT_HOOK

channels = {
    'alert':SLACK_ALERT_HOOK
}

class SlackLogger:

    def __init__(self, channel, runId):
        self.channel = channel
        self.runId = runId
        self.prefix = f"Run ID: *{runId}* "

    def sendAlert(self, text):
        webhook = WebhookClient(url=channels[self.channel])
        if not webhook:
            return False
        response = webhook.send(
            text=self.prefix+text
        )
        return True

    def sendAlertWithMrkdwnBlocks(self, blocks):
        webhook = WebhookClient(url=channels[self.channel])
        if not webhook:
            return False
        response = webhook.send(
            text="\n".join([self.prefix]+blocks),
            blocks=[
                {"type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": self.prefix
                }}
            ]+[{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": b
                }
            } for b in blocks]
        )
        return True