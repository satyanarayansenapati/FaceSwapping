import os
import sys
import logging

# defining the format of the logging string
logging_string = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"

#logging directory
log_dir = 'logs'

log_filepath = os.path.join(log_dir,'running_logs.log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format=logging_string,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('swap')