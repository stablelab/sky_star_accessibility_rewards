
import common.stablelib.slack_interaction.sendAlert as si
from datetime import datetime

def keep_alive(run_id='keepalive'):
    print('keepalive', run_id)
    logger = si.SlackLogger('alert', run_id)
    logger.sendAlertWithMrkdwnBlocks(['*Still Alive*', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

if __name__ == "__main__":
    keep_alive('main-call')