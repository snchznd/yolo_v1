import logging
import datetime
import os

DEFAULT_FORMAT = "[%(asctime)s] - %(levelname)s - | %(message)s"


def get_file_logger(
    logger_name: str,
    log_file: str,
    level: int = logging.DEBUG,
    format: str = DEFAULT_FORMAT,
) -> logging.Logger:
    # create logger
    logger = logging.getLogger(logger_name)
    
    # set logger level
    logger.setLevel(level)
    
    # add handler
    expanded_log_file = os.path.expanduser(log_file)
    full_path = process_file_name(expanded_log_file)
    handler = logging.FileHandler(full_path)
    
    logger.addHandler(handler)
    
    # add formatter
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    
    return logger
    
    
def process_file_name(file_name):
        return file_name + '_' + datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S") + ".log"