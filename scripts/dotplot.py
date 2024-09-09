import os
import pickle

import numpy as np
import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fiiireflyyy.learn as fl

import toml
import logging

with open(os.path.join(os.getcwd(), "config/dotplot.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__dotplot__")


def scatter(accuracies, cups, colormap):
    for r in range(len(accuracies)):
        for c in range(len(accuracies[0])):
            color = cups[r][c]
            dotsize = 300 * 2 ** 4 * accuracies[r][c] / 100 + 20
            print(cups[r][c], dotsize)
            if accuracies[r][c] != 0:
                plt.scatter(x=c, y=r, s=dotsize, c=colormap(color), cmap='Blues', edgecolors='k')


def setup_colormap(fig, ax, colormap):
    colorbar_label = "Confidence Upon Prediction"
    cbar_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cbar_ticklabels = [str(x) for x in cbar_ticks]
    cbar = fig.colorbar(mappable=cm.ScalarMappable(cmap=colormap), ax=ax,
                        shrink=config["figure"]["cbar"]["shrink"], location=config["figure"]["cbar"]["location"],
                        ticks=cbar_ticks,
                        orientation='horizontal',
                        pad=0.05)
    cbar.ax.set_xticklabels(cbar_ticklabels, fontsize=config["figure"]["cbar"]["label_size"])
    cbar.set_label(label=colorbar_label, fontsize=config["figure"]["cbar"]["label_size"], )
    cbar.minorticks_on()


def setup_axes(ax, TRAIN_CORRESPONDENCE, TEST_CORRESPONDENCE):
    ax.set_aspect('equal', )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.gca().invert_yaxis()
    
    ax.set_ylabel(config["figure"]["axes"]["ylabel"], fontdict={"font": config["figure"]["font"],
                                                                "fontsize": config["figure"]["axes"]["ysize"]})
    ax.set_xlabel(config["figure"]["axes"]["xlabel"], fontdict={"font": config["figure"]["font"],
                                                                "fontsize": config["figure"]["axes"]["xsize"]})
    
    if config["figure"]["axes"]["split_labels"]:
        new_test_labels = []
        new_train_labels = []
        test_temp = {}
        train_temp = {}
        
        for test in config["model"]["test"]:
            splits = test.split(config["figure"]["axes"]["split_labels"])
            n_splits = len(splits)
            if n_splits:
                split_pos = int(np.floor(n_splits / 2))
                new_key = (config["figure"]["axes"]["split_labels"].join(splits[:split_pos])
                           + "\n" +
                           config["figure"]["axes"]["split_labels"].join(splits[split_pos:]))
                test_temp[new_key] = TEST_CORRESPONDENCE[test]
                new_test_labels.append(new_key)
        
        for train in config["model"]["train"]:
            splits = train.split(config["figure"]["axes"]["split_labels"])
            n_splits = len(splits)
            if n_splits:
                split_pos = int(np.floor(n_splits / 2))
                new_key = (config["figure"]["axes"]["split_labels"].join(splits[:split_pos])
                           + "\n" +
                           config["figure"]["axes"]["split_labels"].join(splits[split_pos:]))
                train_temp[new_key] = TRAIN_CORRESPONDENCE[train]
                new_train_labels.append(new_key)
        
        ax.set_xticks([test_temp[x] for x in new_test_labels], new_test_labels,
                      fontsize=config["figure"]["ticks"]["xsize"], rotation=config["figure"]["ticks"]["xrot"])
        ax.set_yticks([train_temp[x] for x in new_train_labels], new_train_labels,
                      fontsize=config["figure"]["ticks"]["ysize"], rotation=config["figure"]["ticks"]["yrot"])
    else:
        ax.set_xticks([TEST_CORRESPONDENCE[x] for x in config["model"]["test"]], config["model"]["test"],
                      fontsize=config["figure"]["ticks"]["xsize"], rotation=config["figure"]["ticks"]["xrot"])
        ax.set_yticks([TRAIN_CORRESPONDENCE[x] for x in config["model"]["train"]], config["model"]["train"],
                      fontsize=config["figure"]["ticks"]["ysize"], rotation=config["figure"]["ticks"]["yrot"])


def draw():
    check_params()
    logger.info("Setting up confusion")
    df = pd.read_csv(config["dataset"]["path"], index_col=False)
    df = df[df[config["dataset"]["target_column"]].isin(config["model"]["test"])]
    clf = pickle.load(open(config["model"]["path"], "rb"))
    
    logger.info("Computing confusion")
    df_acc, df_cup, TRAIN_CORRESPONDENCE, TEST_CORRESPONDENCE \
        = fl.test_clf_by_confusion(clf, df, training_targets=config["model"]["train"],
                                   testing_targets=config["model"]["test"],
                                   show=False,
                                   iterations=config["figure"]["iterations"],
                                   return_data=True,
                                   mode='percent', )
    
    if config["figure"]["export_data"]:
        df_acc.to_csv(config["figure"]["export_data"].replace(".csv", "_SCORE.csv"))
        df_cup.to_csv(config["figure"]["export_data"].replace(".csv", "_CUP.csv"))
    
    plt.close()
    fig, ax = plt.subplots(figsize=(1.75 * len(config["model"]["test"]), 1.75 * len(config["model"]["train"])))
    
    accuracies = df_acc.to_numpy().astype(float)
    cups = df_cup.to_numpy().astype(float)
    colormap = plt.get_cmap('Blues')
    
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax.set_axisbelow(True)
    if config["figure"]["grid"]:
        plt.grid(which='major', color='gray', linestyle='dashed')
    if config["figure"]["minor_grid"]:
        plt.grid(which='minor', color='lightgray', linestyle=(0, (5, 10)))
    
    scatter(accuracies, cups, colormap)
    setup_colormap(fig, ax, colormap)
    
    plt.xlim(-0.5, len(df_acc.columns)-0.5)
    plt.ylim(-0.5, len(df_acc)-0.5)
    
    setup_axes(ax, TRAIN_CORRESPONDENCE, TEST_CORRESPONDENCE)
    
    plt.title(config["figure"]["title"]["label"],
              fontdict={'font': config["figure"]["font"],
                        'size': config["figure"]["title"]["size"]})
    fig.tight_layout()
    # fig.set_tight_layout(True)
    if config["figure"]["save"]:
        logger.info(f"Saving figure at : \'{config['figure']['save']}\'")
        plt.savefig(config["figure"]["save"], dpi=config["figure"]["dpi"])
    if config["figure"]["show"]:
        logger.info("Plotting result")
        plt.show()
    
    plt.close()


def check_params():
    if not config['model']['path']:
        raise ValueError('toml: no model loaded, path is emtpy.')
    
    if not config['dataset']['path']:
        raise ValueError('toml: no dataset loaded, path is emtpy.')
    
    if not config['dataset']['target_column']:
        raise ValueError('toml: no target column.')
