from crontab import CronTab

def schedule_runs():
    print('run scheduling')
    cron = CronTab(user='root')
    job = cron.new(command='/usr/local/bin/python app/ping_test.py')
    job.minute.every(2)
    cron.write()
    print('run scheduled', job)