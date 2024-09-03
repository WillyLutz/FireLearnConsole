import logging
import os

import numpy as np
import pandas as pd
import toml
from fiiireflyyy.learn import confidence_ellipse
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

with open(os.path.join(os.getcwd(), "config/pca.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__pca__")


def fit_pca(dataframe: pd.DataFrame, n_components=3, label_column="label"):
    features = dataframe.loc[:, dataframe.columns != label_column].columns
    x = dataframe.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features
    pca = PCA(n_components=n_components)
    principalComponent = pca.fit_transform(x)
    principal_component_columns = [f"principal component {i + 1}" for i in range(n_components)]
    
    pcdf = pd.DataFrame(data=principalComponent
                        , columns=principal_component_columns, )
    pcdf.reset_index(drop=True, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    
    pcdf[label_column] = dataframe[label_column]
    
    return pca, pcdf, pca.explained_variance_ratio_


def apply_pca(pca, dataframe, label_column="label"):
    features = dataframe.loc[:, dataframe.columns != label_column].columns
    x = dataframe.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features
    transformed_ds = pca.transform(x)
    transformed_df = pd.DataFrame(data=transformed_ds,
                                  columns=[f"principal component {i + 1}" for i in range(transformed_ds.shape[1])])
    
    transformed_df.reset_index(drop=True, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    transformed_df[label_column] = dataframe[label_column]
    return transformed_df


def draw():
    if config["pca"]["n_components"] == 2 or config["pca"]["n_components"] == 3:
        pca()

def pca():
    if config["pca"]["n_components"] == 2:
        pca2D()
    else:
        pca3D()


def set_legend(ax):
    if config["figure"]["legend"]["enable"]:
        if config["figure"]["legend"]["anchor"] == 'custom':
            ax.legend(loc='upper left',
                      bbox_to_anchor=(config["figure"]["legend"]["xpos"],
                                      config["figure"]["legend"]["ypos"]),
                      draggable=config["figure"]["legend"]["draggable"],
                      ncols=config["figure"]["legend"]["n_cols"],
                      fontsize=config["figure"]["legend"]["size"],
                      framealpha=config["figure"]["legend"]["framealpha"],
                      )
        else:
            ax.legend(loc=config["figure"]["legend"]["anchor"],
                      draggable=config["figure"]["legend"]["draggable"],
                      ncols=config["figure"]["legend"]["n_cols"],
                      fontsize=config["figure"]["legend"]["size"],
                      framealpha=config["figure"]["legend"]["framealpha"],
                      )
        
        for t, lh in zip(ax.get_legend().texts, ax.get_legend().legend_handles):
            t.set_alpha(config["figure"]["legend"]["alpha"])
            lh.set_alpha(config["figure"]["legend"]["alpha"])
    
    elif ax.get_legend():
        ax.get_legend().remove()
    
    return ax


def setup_pca():
    plt.close()
    logger.info("Setting up")
    fig, ax = plt.subplots(figsize=(config["figure"]["width"], config["figure"]["height"]))
    n_labels = len(config["pca"]["apply"])
    logger.debug(f"n labels = {n_labels}")
    
    df = pd.read_csv(config["pca"]["dataset"], index_col=False)
    
    label_column = config["pca"]["target_column"]
    
    # ---- FIT AND APPLY PCA
    labels_to_fit = config["pca"]["fit"]
    labels_to_apply = config["pca"]["apply"]
    
    df_fit = df[df[label_column].isin(labels_to_fit)]
    n_components = config["pca"]["n_components"]
    pca, pcdf_fit, ratio = fit_pca(df_fit, n_components=n_components, label_column=label_column)
    df_apply = df[df[label_column].isin(labels_to_apply)]
    pcdf_applied = apply_pca(pca, df_apply, label_column=label_column)
    
    ratio = [round(x * 100, 2) for x in ratio]
    
    return fig, ax, pcdf_applied, ratio


def setup_ticks(ax, all_xmin, all_xmax, all_ymin, all_ymax, all_zmin=None, all_zmax=None):
    xmin = min(all_xmin)
    xmax = max(all_xmax)
    xstep = (xmax - xmin) / (config["figure"]["ticks"]["nx"] - 1)
    xtick = xmin
    xticks = []
    for i in range(config["figure"]["ticks"]["nx"] - 1):
        xticks.append(xtick)
        xtick += xstep
    xticks.append(xmax)
    rounded_xticks = list(np.around(np.array(xticks), config["figure"]["ticks"]["xround"]))
    ax.set_xticks(rounded_xticks)
    ax.tick_params(axis='x',
                   labelsize=config["figure"]["ticks"]["xsize"],
                   labelrotation=config["figure"]["ticks"]["xrot"])
    
    ymin = min(all_ymin)
    ymax = max(all_ymax)
    ystep = (ymax - ymin) / (config["figure"]["ticks"]["ny"] - 1)
    ytick = ymin
    yticks = []
    for i in range(config["figure"]["ticks"]["ny"] - 1):
        yticks.append(ytick)
        ytick += ystep
    yticks.append(ymax)
    rounded_yticks = list(np.around(np.array(yticks), config["figure"]["ticks"]["yround"]))
    ax.set_yticks(rounded_yticks)
    ax.tick_params(axis='y',
                   labelsize=config["figure"]["ticks"]["ysize"],
                   labelrotation=config["figure"]["ticks"]["yrot"])
    
    if config["pca"]["n_components"] == 3:
        zmin = min(all_zmin)
        zmax = max(all_zmax)
        zstep = (zmax - zmin) / (config["figure"]["ticks"]["nz"] - 1)
        ztick = zmin
        zticks = []
        for i in range(config["figure"]["ticks"]["nz"] - 1):
            zticks.append(ztick)
            ztick += zstep
        zticks.append(zmax)
        rounded_zticks = list(np.around(np.array(zticks), config["figure"]["ticks"]["zround"]))
        ax.set_zticks(rounded_zticks)
        ax.tick_params(axis='z',
                       labelsize=config["figure"]["ticks"]["zsize"],
                       labelrotation=config["figure"]["ticks"]["zrot"])
    
    return ax


def setup_labels(ax, ratio):
    show_ratiox = f' ({ratio[0]}%)' if config["pca"]["show_ratio"] else ''
    show_ratioy = f' ({ratio[1]}%)' if config["pca"]["show_ratio"] else ''
    ax.set_xlabel(config["figure"]["axes"]["xlabel"] + show_ratiox,
                  fontdict={"font": config["figure"]["font"],
                            "fontsize": config["figure"]["axes"]["xsize"]})
    ax.set_ylabel(config["figure"]["axes"]["ylabel"] + show_ratioy,
                  fontdict={"font": config["figure"]["font"],
                            "fontsize": config["figure"]["axes"]["ysize"]})
    
    if config["pca"]["n_components"] == 3:
        show_ratioz = f' ({ratio[2]}%)' if config["pca"]["show_ratio"] else ''
        ax.set_zlabel(config["figure"]["axes"]["zlabel"] + show_ratioz,
                      fontdict={"font": config["figure"]["font"],
                                "fontsize": config["figure"]["axes"]["zsize"]})
    
    ax.set_title(config["figure"]["title"]["label"],
                 fontdict={"font": config["figure"]["font"],
                           "fontsize": config["figure"]["title"]["size"]}, )
    
    return ax


def pca2D():
    check_params()
    labels_to_apply = config["pca"]["apply"]
    label_column = config["pca"]["target_column"]
    
    fig, ax, pcdf_applied, ratio = setup_pca()
    # ----- PLOTTING
    all_ymin = []  # for ticks
    all_ymax = []
    all_xmin = []
    all_xmax = []
    for l in range(len(labels_to_apply)):
        label = labels_to_apply[l]
        x_data = pcdf_applied.loc[pcdf_applied[label_column] == label][pcdf_applied.columns[0]]
        all_xmax.append(max(x_data))
        all_xmin.append(min(x_data))
        y_data = pcdf_applied.loc[pcdf_applied[label_column] == label][pcdf_applied.columns[1]]
        all_ymin.append(min(y_data))
        all_ymax.append(max(y_data))
        
        ax.scatter(x_data, y_data,
                   s=config["figure"]["marker_size"],
                   marker=config["figure"]["marker"],
                   color=config["figure"]["colors"][l],
                   # todo known exception : len(colors) >= len(labels to apply)
                   alpha=float(config["figure"]["alphas"][l]),  # todo same here
                   label=label
                   )
        if config["pca"]["ellipsis"]:
            ax.scatter(np.mean(x_data), np.mean(y_data),
                       marker="+",
                       color=config["figure"]["colors"][l],
                       alpha=float(config["figure"]["alphas"][l]),
                       linewidth=2,
                       s=160)
            confidence_ellipse(x_data, y_data, ax, n_std=1.0,
                               alpha=float(config["figure"]["alphas"][l]),
                               color=config["figure"]["colors"][l],
                               fill=False, linewidth=2)
    
    # ---- LABELS
    ax = setup_labels(ax=ax, ratio=ratio)
    
    # ---- TICKS
    ax = setup_ticks(ax=ax,
                     all_xmin=all_xmin,
                     all_xmax=all_xmax,
                     all_ymin=all_ymin,
                     all_ymax=all_ymax)
    
    # ----- LEGEND
    ax = set_legend(ax=ax)
    
    plt.tight_layout()
    if config["figure"]["save"]:
        logger.info(f"Saving figure at : \'{config['figure']['save']}\'")
        plt.savefig(config["figure"]["save"], dpi=config["figure"]["dpi"])
    if config["figure"]["show"]:
        logger.info("Plotting result")
        plt.show()
    
    plt.close()


def pca3D():
    check_params()
    logger.info("Drawing PCA 3D")
    
    labels_to_apply = config["pca"]["apply"]
    label_column = config["pca"]["target_column"]
    
    fig, ax, pcdf_applied, ratio = setup_pca()
    # ----- PLOTTING
    all_ymin = []  # for ticks
    all_ymax = []
    all_xmin = []
    all_xmax = []
    all_zmin = []
    all_zmax = []
    
    ax = plt.axes(projection='3d')
    for l in range(len(labels_to_apply)):
        label = labels_to_apply[l]
        x_data = pcdf_applied.loc[pcdf_applied[label_column] == label][pcdf_applied.columns[0]]
        all_xmax.append(max(x_data))
        all_xmin.append(min(x_data))
        y_data = pcdf_applied.loc[pcdf_applied[label_column] == label][pcdf_applied.columns[1]]
        all_ymin.append(min(y_data))
        all_ymax.append(max(y_data))
        z_data = pcdf_applied.loc[pcdf_applied[label_column] == label][pcdf_applied.columns[2]]
        all_zmin.append(min(z_data))
        all_zmax.append(max(z_data))
        
        ax.scatter3D(x_data, y_data, z_data,
                     s=config["figure"]["marker_size"],
                     marker=config["figure"]["marker"],
                     color=config["figure"]["colors"][l],
                     # todo known exception : len(colors) >= len(labels to apply)
                     alpha=config["figure"]["alphas"][l],  # todo same here
                     label=label
                     )
        
        # ---- LABELS
    ax = setup_labels(ax=ax, ratio=ratio)
    
    # ---- TICKS
    ax = setup_ticks(ax=ax,
                     all_xmin=all_xmin,
                     all_xmax=all_xmax,
                     all_ymin=all_ymin,
                     all_ymax=all_ymax,
                     all_zmin=all_zmin,
                     all_zmax=all_zmax)
    
    # ----- LEGEND
    ax = set_legend(ax=ax)

    
    plt.tight_layout()
    if config["figure"]["save"]:
        logger.info(f"Saving figure at : \'{config['figure']['save']}\'")
        plt.savefig(config["figure"]["save"], dpi=config["figure"]["dpi"])
    if config["figure"]["show"]:
        logger.info("Plotting result")
        plt.show()
    
    plt.close()


def check_params():
    if not config['pca']['dataset']:
        raise ValueError("toml: 'path' is missing, no dataset loaded.")
    
    if not len(config['figure']['colors']) >= len(config['pca']['apply']):
        raise ValueError("toml: not enough colors for the number of targets")
    
    if not len(config['figure']['alphas']) >= len(config['pca']['apply']):
        raise ValueError("toml: not enough alphas for the number of targets")
    
    if not config['pca']['apply']:
        raise ValueError("toml: 'apply' is empty. At least one label is required.")
    
    if not config['pca']['fit']:
        raise ValueError("toml: 'fit' is empty. At least one label is required.")



