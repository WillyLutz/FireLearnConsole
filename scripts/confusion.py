import os
import pickle

import numpy as np
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
                                   mode=config["figure"]["mode"],)
    
    if config["figure"]["export_data"]:
        df_acc.to_csv(config["figure"]["export_data"].replace(".csv", "_SCORE.csv"))
        df_cup.to_csv(config["figure"]["export_data"].replace(".csv", "_CUP.csv"))
    
    mixed_labels_matrix = np.empty((len(TRAIN_CORRESPONDENCE.keys()), len(TEST_CORRESPONDENCE.keys()))).tolist()
    
    acc_array = df_acc.to_numpy().astype(float) if config["figure"]["mode"] == 'percent' else df_acc.to_numpy().astype(int)
    cup_array = df_cup.to_numpy()
    for r in range(len(acc_array)):
        for c in range(len(acc_array[0])):
            case = f"{acc_array[r][c]}%\nCUP={cup_array[r][c]}" if config["figure"]["mode"] == 'percent' else f"{acc_array[r][c]}\nCUP={cup_array[r][c]}"
            mixed_labels_matrix[r][c] = case
    plt.close()
    fig, ax = plt.subplots(figsize=(config["figure"]["width"], config["figure"]["height"]))
    sns.heatmap(ax=ax, data=acc_array, annot=mixed_labels_matrix,
                annot_kws=config["figure"]["annot"], fmt='', cmap="Blues",
                square=True, cbar_kws=config["figure"]["cbar"])
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
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
        
        ax.set_xticks([test_temp[x] + 0.5 for x in new_test_labels], new_test_labels,
                      fontsize=config["figure"]["ticks"]["xsize"], rotation=config["figure"]["ticks"]["xrot"])
        ax.set_yticks([train_temp[x] + 0.5 for x in new_train_labels], new_train_labels,
                      fontsize=config["figure"]["ticks"]["ysize"], rotation=config["figure"]["ticks"]["yrot"])
    else:
        ax.set_xticks([TEST_CORRESPONDENCE[x] + 0.5 for x in config["model"]["test"]], config["model"]["test"],
                      fontsize=config["figure"]["ticks"]["xsize"], rotation=config["figure"]["ticks"]["xrot"])
        ax.set_yticks([TRAIN_CORRESPONDENCE[x] + 0.5 for x in config["model"]["train"]], config["model"]["train"],
                      fontsize=config["figure"]["ticks"]["ysize"], rotation=config["figure"]["ticks"]["yrot"])
    
    plt.title(config["figure"]["title"]["label"],
              fontdict={'font': config["figure"]["font"],
                        'size': config["figure"]["title"]["size"]})
    plt.tight_layout()
    
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
