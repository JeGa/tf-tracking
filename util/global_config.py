import yaml
import logging
import os

cfg = None


def read(config_file):
    logging.info('Read configuration file ' + config_file + '.')

    with open(config_file, 'r') as ymlfile:
        global cfg
        cfg = yaml.load(ymlfile)

    create_directories()


def create_directories():
    create_dir(cfg['train_summaries'])
    create_dir(cfg['checkpoints'])
    create_dir(cfg['results'])


def create_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
