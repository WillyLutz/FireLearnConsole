import os
import pickle

import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt

import fiiireflyyy.learn as fl

import toml
import logging

with open(os.path.join(os.getcwd(), "config/confusion.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__confusion__")


def draw():
    logger.info("Setting up confusion")
    df = pd.read_csv(config["dataset"]["path"], index_col=False)
    df = df[df[config["dataset"]["target_column"]].isin(config["model"]["test"])]
    clf = pickle.load(open(config["model"]["path"], "rb"))
    
    logger.info("Computing confusion")
    overall_matrix, mixed_labels_matrix, TRAIN_CORRESPONDENCE, TEST_CORRESPONDENCE \
        = fl.test_clf_by_confusion(clf, df, training_targets=config["model"]["train"],
                                   testing_targets=config["model"]["test"],
                                   show=config["figure"]["show"],
                                   iterations=config["figure"]["iterations"],
                                   return_data=True,
                                   mode=config["figure"]["mode"])
    
    plt.close()
    fig, ax = plt.subplots(figsize=(config["figure"]["width"], config["figure"]["height"]))
    sns.heatmap(ax=ax, data=overall_matrix, annot=mixed_labels_matrix,
                annot_kws=config["figure"]["annot"], fmt='', cmap="Blues",
                square=True, cbar_kws=config["figure"]["cbar"])
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_ylabel(config["figure"]["axes"]["ylabel"], fontdict={"font": config["figure"]["font"],
                                                                "fontsize": config["figure"]["axes"]["ysize"]})
    ax.set_xlabel(config["figure"]["axes"]["xlabel"], fontdict={"font": config["figure"]["font"],
                                                                "fontsize": config["figure"]["axes"]["xsize"]})
    
    ax.set_xticks([TEST_CORRESPONDENCE[x] + 0.5 for x in config["model"]["test"]], config["model"]["test"],
                  fontsize=config["figure"]["ticks"]["xsize"], rotation=config["figure"]["ticks"]["xrot"])
    ax.set_yticks([TRAIN_CORRESPONDENCE[x] + 0.5 for x in config["model"]["train"]], config["model"]["train"],
                  fontsize=config["figure"]["ticks"]["ysize"], rotation=config["figure"]["ticks"]["yrot"])
    
    plt.title(config["figure"]["title"]["label"],
              fontdict={'font': config["figure"]["font"],
                        'size': config["figure"]["size"]})
    plt.tight_layout()
    
    if config["figure"]["save"]:
        logger.info(f"Saving figure at : \'{config['figure']['save']}\'")
        plt.savefig(config["figure"]["save"], dpi=config["figure"]["dpi"])
    if config["figure"]["show"]:
        logger.info("Plotting result")
        plt.show()
    
    plt.close()
    fl.plot_pca()
