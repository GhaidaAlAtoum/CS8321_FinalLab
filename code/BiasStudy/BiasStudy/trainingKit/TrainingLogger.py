import logging
from pathlib import Path
import sys

def setup_logger(
    logger_name: str, 
    output_dir: str
):
    formatter = logging.Formatter(fmt='------------ %(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logging_output_dir = get_logger_outputdir(output_dir)
    
    logging_output_file_path = "{}/{}.log".format(
        logging_output_dir,
        logger_name
    )
    
    handler = logging.FileHandler(logging_output_file_path, mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

def get_logger_outputdir(output_dir: str):
    logging_output_dir = "{}/logs".format(output_dir)
    Path(logging_output_dir).mkdir(parents=True, exist_ok=True)
    return logging_output_dir