import os
import sys
import logging
import toml

from scripts import processing, learning, confusion, feature_importance, pca, simple_plot



with open(os.path.join(os.getcwd(), "config/firelearn.toml")) as f:
    config = toml.load(f)

if config["logging"]["clear_each_run"] is True:
    with open(os.path.join(os.getcwd(), config["logging"]["out"]), 'w'):
        pass

logger = logging.getLogger("__main__")
logging.basicConfig(filename="firelearn.out", level=logging.DEBUG,
                    format=config["logging"]["format"],
                    datefmt=config["logging"]["datefmt"])
args = sys.argv
if __name__ == '__main__':
    logger.debug(args)
    
    if "-p" in args:
        processing.process()
        
    if '-l' in args:
        learning.learn()
        
    if '-c' in args:
        confusion.draw()
        
    if '-i' in args:
        feature_importance.draw()
        
    if "-pca" in args:
        pca.draw()
        
    if "-plot" in args:
        simple_plot.draw()