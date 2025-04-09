# Use a lightweight Python 3 base image
FROM python:3.10-bookworm

# Install system dependencies needed by psycopg2 (if required)
RUN apt-get update && apt-get install -y \
    # cron \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get install --no-install-recommends -y cron 

# ADD cron.sh /usr/bin/cron.sh
# RUN chmod +x /usr/bin/cron.sh

# ADD ./crontab /etc/cron.d/cron-jobs
# RUN chmod 0644 /etc/cron.d/cron-jobs

# RUN touch /var/log/cron.log
# RUN touch /var/log/cron.log && \
#     touch /var/log/cron-status.log && \
#     chmod 0644 /var/log/cron.log && \
#     chmod 0644 /var/log/cron-status.log
# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY code/ ./

# Expose port 8080 for Cloud Run
ENV PORT=8080

RUN ls -la

# RUN tail -f /var/log/syslog | grep CRON

# Start the server (using Uvicorn as an example)
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["python", "simple_server.py"]

# Add script to check and log cron service status
# RUN echo '#!/bin/bash\n\
# service cron status >> /var/log/cron-status.log 2>&1\n\
# service cron start >> /var/log/cron-status.log 2>&1\n\
# python simple_server.py\n\
# ' > /app/start.sh && chmod +x /app/start.sh

# RUN echo '*/5 * * * * root /usr/local/bin/python /app/start.sh >> /var/log/cron.log 2>&1\n' > /etc/cron.d/cron-jobs
# RUN crontab -l
RUN whoami
RUN which python

# CMD ["/app/start.sh"]
CMD ["python", "simple_server.py"]
