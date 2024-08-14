import os
import pickle
import toml
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scripts.ClfTester import ClfTester

with open(os.path.join(os.getcwd(), "config/learn.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__learn__")


def label_encoding(y):
    labels = list(set(list(y)))
    corr = {}
    for lab in range(len(labels)):
        corr[labels[lab]] = lab
    
    for key, value in corr.items():
        y.replace(key, value)
    
    return y


def learning_display_computed_metrics(metrics_elements, all_train_metrics,
                                      all_test_metrics,
                                      all_train_scores, all_test_scores, ):
    metrics_elements.append("CLASSIFICATION METRICS")
    metrics_elements.append("---------------------------------------------------------------")
    metrics_elements.append("")
    
    pm = u"\u00B1"
    if config["model"]["load"]:  # ALREADY TRAINED
        metrics_elements.append(f"Number of training iterations: None (pre-trained classifier)")
    else:
        metrics_elements.append(f"Number of training iterations: {int(config['model']['train']['n_iter'])}", )
    metrics_elements.append(f"Number of testing iterations: {int(config['model']['train']['n_iter'])}", )
    
    if config["model"]["load"]:  # ALREADY TRAINED
        metrics_elements.append(f"Training accuracy: None (pre-trained classifier)")
    else:
        metrics_elements.append(
            f"Training accuracy: {str(np.mean(all_train_scores).round(3))} {pm} {str(np.std(all_train_scores).round(3))}", )
    metrics_elements.append(
        f"Testing accuracy: {str(np.mean(all_test_scores).round(3))} {pm} {str(np.std(all_test_scores).round(3))}")
    
    if config["model"]["test"]["metrics"]:
        
        metrics_elements.append("")
        metrics_elements.append("TRAINING------------------------")
        if not config["model"]["load"]:  # not pre-trained:
            for t in config["model"]["train"]["targets"]:
                t_metrics = {t: [[], []]}  # {class1 : [[true probas],[false probas]] }
                
                for train_metrics in all_train_metrics:
                    for target, target_metric in train_metrics.items():
                        if t == target:
                            t_metrics[t][0] = t_metrics[t][0] + target_metric[0]
                            t_metrics[t][1] = t_metrics[t][1] + target_metric[1]
                
                true_preds = t_metrics[t][0]
                false_preds = t_metrics[t][1]
                if not true_preds:
                    true_preds.append(0)
                if not false_preds:
                    false_preds.append(0)
                metrics_elements.append("---")
                metrics_elements.append(f"Target: {t}")
                metrics_elements.append(f"Number of true predictions: {len(true_preds)}")
                metrics_elements.append(f"Number of false predictions: {len(false_preds)}")
                metrics_elements.append(
                    f"CUP for true predictions: {np.mean(true_preds).round(3)} {pm} {np.std(true_preds).round(3)}")
                metrics_elements.append(
                    f"CUP for false predictions: {np.mean(false_preds).round(3)} {pm} {np.std(false_preds).round(3)}")
                metrics_elements.append(f"")
        else:
            metrics_elements.append("No data available - Classifier were pre-trained.")
        
        metrics_elements.append("")
        metrics_elements.append("TESTING------------------------")
        for t in config["model"]["train"]["targets"]:
            t_metrics = {t: [[], []]}  # {class1 : [[true probas],[false probas]] }
            
            for test_metrics in all_test_metrics:
                for target, target_metric in test_metrics.items():
                    if t == target:
                        t_metrics[t][0] = t_metrics[t][0] + target_metric[0]
                        t_metrics[t][1] = t_metrics[t][1] + target_metric[1]
            
            true_preds = t_metrics[t][0]
            false_preds = t_metrics[t][1]
            if not true_preds:
                true_preds.append(0)
            if not false_preds:
                false_preds.append(0)
            metrics_elements.append("---")
            metrics_elements.append(f"Target: {t}")
            metrics_elements.append(f"Number of true predictions: {len(true_preds)}")
            metrics_elements.append(f"Number of false predictions: {len(false_preds)}")
            metrics_elements.append(
                f"CUP for true predictions: {np.mean(true_preds).round(3)} {pm} {np.std(true_preds).round(3)}")
            metrics_elements.append(
                f"CUP for false predictions: {np.mean(false_preds).round(3)} {pm} {np.std(false_preds).round(3)}")
            metrics_elements.append(f"")
    else:
        metrics_elements.append("-------------")
        metrics_elements.append("No additional metrics computed.")
    
    return metrics_elements


def save_object(obj, path):
    file = open(f'{path}', 'wb')
    pickle.dump(obj, file)
    file.close()

def setup_learn():
    logger.info("fetching parameters")
    rfc_params = config["model"]["params"]["rfc"]
    logger.debug(f"Model parameters : {rfc_params}")
    train_df = pd.read_csv(config["dataset"]["train"], index_col=False)
    test_df = pd.read_csv(config["dataset"]["test"], index_col=False)
    
    target_column = config["dataset"]["target_column"]
    
    logger.info("Fetching data")
    train_df = train_df[train_df[target_column].isin(config["model"]["train"]["targets"])]
    train_df.reset_index(inplace=True, drop=True)
    y_train = train_df[target_column]
    y_train = label_encoding(y_train)
    X_train = train_df.loc[:, train_df.columns != target_column]
    
    test_df = test_df[test_df[target_column].isin(config["model"]["train"]["targets"])]
    test_df.reset_index(inplace=True, drop=True)
    y_test = test_df[target_column]
    y_test = label_encoding(y_test)
    X_test = test_df.loc[:, test_df.columns != target_column]
    
    
    logger.info("Setting up model")
    rfc = RandomForestClassifier()
    rfc.set_params(**rfc_params)
    
    return rfc, X_train, y_train, X_test, y_test

def learn():
    all_test_scores = []
    all_train_scores = []
    all_train_metrics = []
    all_test_metrics = []
    
    if config['dataset']['split']:
        split_dataset()
        
        
    rfc, X_train, y_train, X_test, y_test = setup_learn()
    logger.info("Training initiated")
    for iteration in range(int(config["model"]["train"]["n_iter"])):
        clf_tester = ClfTester(rfc)
        if config["model"]["load"]:
            clf_tester.trained = True
        
        clf_tester.train(X_train, y_train)
        clf_tester.test(X_test, y_test)
        
        if not clf_tester.trained:
            all_train_scores.append(clf_tester.train_acc)
            all_train_metrics.append(clf_tester.train_metrics)
        all_test_scores.append(clf_tester.test_acc)
        all_test_metrics.append(clf_tester.test_metrics)
        
    logger.info("Displaying results")
    metrics_elements = []
    metrics_elements = learning_display_computed_metrics(metrics_elements,
                                                         all_train_metrics,
                                                         all_test_metrics, all_train_scores,
                                                         all_test_scores)
    for x in metrics_elements:
        print(x)
    
    if config["model"]["save_model"]:
        save_object(rfc, config["model"]["save_model"])
        
    if config["model"]["save_metrics"]:
        with open(config["model"]["save_metrics"], "w") as f:
            for line in metrics_elements:
                f.write(line+"\n")
        

def split_dataset():
    logger.info(f"Splitting dataset {config['dataset']['split']}")
    ratio = config["dataset"]["ratio"]
    path = config['dataset']['split']

    if path:
        df = pd.read_csv(path)
        train_sets = []
        test_sets = []
        labels = list(set(list(df["label"])))
        for label in labels:
            dfl = df[df["label"] == label]
            dfl.reset_index(inplace=True, drop=True)
            y = dfl["label"]
            # y = self.label_encoding(y)
            X = dfl.loc[:, df.columns != "label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio)
            X_train["label"] = y_train
            X_test["label"] = y_test
            train_sets.append(X_train)
            test_sets.append(X_test)

        train_df = pd.concat(train_sets, ignore_index=True)
        train_df.reset_index(inplace=True, drop=True)
        test_df = pd.concat(test_sets, ignore_index=True)
        test_df.reset_index(inplace=True, drop=True)

        base_path = path.split(".")
        train_df.to_csv(base_path[0]+"_Xy_train." + base_path[1], index=False)
        test_df.to_csv(base_path[0] + "_Xy_test." + base_path[1], index=False)

        logger.info("Splitting done")
