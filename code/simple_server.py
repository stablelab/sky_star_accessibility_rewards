from http.server import HTTPServer, BaseHTTPRequestHandler

import schedule
import time

import ping_test
import schedule_runs
import getpass
import threading
import referral_net_tracker_script_clean
import os
import logging

import common.slogger as slogger

logger = slogger.logger
logger.info('loaded simple_server.py')


def log_active():
    logger.info('active')

def job_that_executes_once():
    logger.info('executing once')
    ping_test.keep_alive('scheduled MANUAL EXECUTION')
    referral_net_tracker_script_clean.call_script()
    time.sleep(10)
    return schedule.CancelJob

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path, self.path)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Hello, World!" + bytes(self.path, 'utf-8'))
        ping_test.keep_alive('manual-ping' + self.path)
        if(self.path == '/start/'+os.getenv('ACCESS_TOKEN_AR')):
            schedule.every().seconds.do(job_that_executes_once)
        else:
            logger.info('invalid access token')

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHandler)
    print(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    logger.info(f"Program is being run by user: {getpass.getuser()}")
    logger.info("SNOWFLAKE USER %s", os.getenv('SNOWFLAKE_USER'))
    ping_test.keep_alive('entry')
    # schedule_runs.schedule_runs()
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    schedule.every().day.at('10:25',"Europe/Amsterdam").do(ping_test.keep_alive)
    schedule.every().day.at("04:30","Europe/Amsterdam").do(referral_net_tracker_script_clean.call_script)
    schedule.every().day.at("11:55","Europe/Amsterdam").do(ping_test.keep_alive)
    schedule.every().day.at("11:50","Europe/Amsterdam").do(referral_net_tracker_script_clean.call_script)
    schedule.every().day.at("14:40","Europe/Amsterdam").do(ping_test.keep_alive)
    schedule.every().day.at("17:30","Europe/Amsterdam").do(referral_net_tracker_script_clean.call_script)
    schedule.every().day.at("22:30","Europe/Amsterdam").do(ping_test.keep_alive)
    logger.info('jobs setup')
    schedule.every(5).minutes.do(log_active)
    while True:
        schedule.run_pending()
        time.sleep(10)