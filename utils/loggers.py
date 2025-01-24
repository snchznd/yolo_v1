import logging
import datetime

LOGGING_PATH = "/home/masn/projects/yolo/logs"
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
    handler = logging.FileHandler(process_file_name(log_file))
    
    logger.addHandler(handler)
    
    # add formatter
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    
    return logger
    
    
def process_file_name(file_name):
        return file_name + '_' + datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S") + ".log"