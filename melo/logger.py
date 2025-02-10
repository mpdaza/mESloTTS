import logging
import os

logger = logging.getLogger(__name__)
# # Definir loggers a nivel de módulo
# log = None
# log_train = None

def init():
    global log, log_train
    log_path =  'logs_personalizados'
    log = get_logger(log_path)
    log_train = get_logger('logs/config')

def get_logger(model_dir, filename="train.log"):
    global logger   
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger