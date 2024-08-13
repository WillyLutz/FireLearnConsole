import os
import toml
import logging
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

with open(os.path.join(os.getcwd(), "config/simple_plot.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__simple_plot__")


def set_ticks(ax, n_val, ymin, ymax):
    x_data = [i for i in range(n_val)]
    xmin = min(x_data)
    xmax = max(x_data)
    xstep = (xmax - xmin) / (config["ticks"]["nx"] - 1)
    xtick = xmin
    xticks = []
    for i in range(config["ticks"]["nx"] - 1):
        xticks.append(xtick)
        xtick += xstep
    xticks.append(xmax)
    rounded_xticks = list(np.around(np.array(xticks), config["ticks"]["xround"]))
    ax.set_xticks(rounded_xticks)
    ax.tick_params(axis='x',
                   labelsize=config["ticks"]["xsize"],
                   labelrotation=config["ticks"]["xrot"])
    
    ystep = (ymax - ymin) / (config["ticks"]["ny"] - 1)
    ytick = ymin
    yticks = []
    for i in range(config["ticks"]["ny"] - 1):
        yticks.append(ytick)
        ytick += ystep
    yticks.append(ymax)
    rounded_yticks = list(np.around(np.array(yticks), config["ticks"]["yround"]))
    ax.set_yticks(rounded_yticks)
    ax.tick_params(axis='y',
                   labelsize=config["ticks"]["ysize"],
                   labelrotation=config["ticks"]["yrot"])


def set_labels(ax):
    ax.set_xlabel(config["axes"]["xlabel"],
                  fontdict={"font": config["figure"]["font"],
                            "fontsize": config["axes"]["xsize"]})
    ax.set_ylabel(config["axes"]["ylabel"],
                  fontdict={"font": config["figure"]["font"],
                            "fontsize": config["axes"]["ysize"]})
    ax.set_title(config["title"]["label"],
                 fontdict={"font": config["figure"]["font"],
                           "fontsize": config["title"]["size"], })


def set_legend(ax):
    if config["legend"]["enable"]:
        if config["legend"]["anchor"] == 'custom':
            ax.legend(loc='upper left',
                      bbox_to_anchor=(config["legend"]["xpos"],
                                      config["legend"]["ypos"]),
                      draggable=config["legend"]["draggable"],
                      ncols=config["legend"]["n_cols"],
                      fontsize=config["legend"]["size"],
                      framealpha=config["legend"]["framealpha"],
                      )
        else:
            ax.legend(loc=config["legend"]["anchor"],
                      draggable=config["legend"]["draggable"],
                      ncols=config["legend"]["n_cols"],
                      fontsize=config["legend"]["size"],
                      framealpha=config["legend"]["framealpha"],
                      )
        
        for t, lh in zip(ax.get_legend().texts, ax.get_legend().legend_handles):
            t.set_alpha(config["legend"]["alpha"])
            lh.set_alpha(config["legend"]["alpha"])
    
    elif ax.get_legend():
        ax.get_legend().remove()


def draw():
    logger.info("Plotting dataset")
    fig, ax = plt.subplots(figsize=(config["figure"]["width"], config["figure"]["height"]))
    
    df = pd.read_csv(config["dataset"]["path"])
    targets = config["dataset"]["targets"]
    n_val = 0
    ymin = 0
    ymax = 0
    for i in range(len(targets)):
        target = targets[i]
        sub = df[df[config["dataset"]["target_column"]] == target]
        sub = sub.reset_index(drop=True)
        sub = sub.loc[:, ~df.columns.isin([config["dataset"]["target_column"], "ID"])]
        ymax = sub.to_numpy().max() if sub.to_numpy().max() > ymax else ymax
        ymin = sub.to_numpy().min() if sub.to_numpy().min() < ymin else ymin
        
        stds = sub.std(axis=0)
        means = sub.mean(axis=0)
        
        ax.plot(means, color=config["figure"]["colors"][i],
                label=target,
                linewidth=config["figure"]["linewidth"],
                linestyle=config["figure"]["linestyle"])
        ax.fill_between([x for x in range(len(means))],
                        [means[x] - stds[x] for x in means.index],
                        [means[x] + stds[x] for x in means.index],
                        color=config["figure"]["colors"][i],
                        alpha=config["figure"]["fillalpha"])
        n_val = len(means)
    
    set_ticks(ax, n_val, ymin, ymax)
    set_labels(ax)
    set_legend(ax)
    plt.tight_layout()
    
    if config["figure"]["save"]:
        logger.info(f"Saving figure at : \'{config['figure']['save']}\'")
        plt.savefig(config["figure"]["save"], dpi=config["figure"]["dpi"])
    if config["figure"]["show"]:
        logger.info("Plotting result")
        plt.show()
    
    # self.view.figures["plot"] = (fig, ax)
    # self.view.canvas["plot"].draw()
