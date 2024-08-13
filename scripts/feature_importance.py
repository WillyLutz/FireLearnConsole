import os
import pickle
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import logging
import toml

with open(os.path.join(os.getcwd(), "config/feature_importance.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__feat_imp__")

def draw():
    logger.info("Setting up model")
    
    clf = pickle.load(open(config["model"]["path"], "rb"))
    
    fig, ax = plt.subplots(figsize=(config["figure"]["width"], config["figure"]["height"]))
    y_data = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    x_data = [i for i in range(len(y_data))]
    
    n_xticks = config["figure"]["ticks"]["nx"]
    n_yticks = config["figure"]["ticks"]["ny"]
    
    if n_yticks > len(y_data):
        logger.error(f"Can not have more y ticks {n_yticks} than y values {len(y_data)}")
        return False
    if n_xticks > len(x_data):
        logger.error(f"Can not have more x ticks {n_xticks} than x values {len(x_data)}")
        return False
    
    logger.info("Plotting")
    ax.plot(x_data, y_data,
            linewidth=config["figure"]["linewidth"],
            linestyle=config["figure"]["linestyle"],
            color=config["figure"]["color"],
            alpha=config["figure"]["alpha"],
            )
    
    fill = config["figure"]["fill"]
    if fill != 'none':
        ylim = plt.gca().get_ylim()
        if fill == 'below':
            ax.fill_between(x_data, y_data, ylim[0],
                            color=config["figure"]["color"],
                            alpha=config["figure"]["alpha_fill"]
                            )
        if fill == 'above':
            ax.fill_between(x_data, y_data, ylim[1],
                            color=config["figure"]["color"],
                            alpha=config["figure"]["alpha_fill"]
                            )
    
    ax.set_xlabel(config["figure"]["axes"]["xlabel"],
                  fontdict={"font": config["figure"]["font"],
                            "fontsize": config["figure"]["axes"]["xsize"]})
    ax.set_ylabel(config["figure"]["axes"]["ylabel"],
                  fontdict={"font": config["figure"]["font"],
                            "fontsize": config["figure"]["axes"]["ysize"]})
    ax.set_title(config["figure"]["title"]["label"],
                 fontdict={"font": config["figure"]["font"],
                           "fontsize": config["figure"]["title"]["size"], })
    xmin = min(x_data)
    xmax = max(x_data)
    xstep = (xmax - xmin) / (n_xticks - 1)
    xtick = xmin
    xticks = []
    for i in range(n_xticks - 1):
        xticks.append(xtick)
        xtick += xstep
    xticks.append(xmax)
    rounded_xticks = list(np.around(np.array(xticks), config["figure"]["ticks"]["xround"]))
    ax.set_xticks(rounded_xticks)
    ax.tick_params(axis='x',
                   labelsize=config["figure"]["ticks"]["xsize"],
                   labelrotation=config["figure"]["ticks"]["xrot"])
    
    ymin = min(y_data)
    ymax = max(y_data)
    ystep = (ymax - ymin) / (n_yticks - 1)
    ytick = ymin
    yticks = []
    for i in range(n_yticks - 1):
        yticks.append(ytick)
        ytick += ystep
    yticks.append(ymax)
    rounded_yticks = list(np.around(np.array(yticks), config["figure"]["ticks"]["yround"]))
    ax.set_yticks(rounded_yticks)
    ax.tick_params(axis='y',
                   labelsize=config["figure"]["ticks"]["ysize"],
                   labelrotation=config["figure"]["ticks"]["yrot"])
    
    plt.tight_layout()
    if config["figure"]["save"]:
        logger.info(f"Saving figure at : \'{config['figure']['save']}\'")
        plt.savefig(config["figure"]["save"], dpi=config["figure"]["dpi"])
    if config["figure"]["show"]:
        logger.info("Plotting result")
        plt.show()
    
    plt.close()