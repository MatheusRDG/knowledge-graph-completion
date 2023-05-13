""" Utilitary functions for the project """

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json


# Load data
def load_fb15k237(path: str):
    """
    Load FB15k-237 dataset
    :param path: path to the dataset
    :return: train, valid, test, entity2wikidata
    """
    columns = ["head", "relation", "tail"]
    train = pd.read_csv(path + "/train.txt", sep="\t", header=None, names=columns)
    valid = pd.read_csv(path + "/valid.txt", sep="\t", header=None, names=columns)
    test = pd.read_csv(path + "/test.txt", sep="\t", header=None, names=columns)
    entity2wikidata = json.load(open(path + "/entity2wikidata.json"))

    return train, valid, test, entity2wikidata


def load_wn18rr(path: str):
    """
    Load WN18RR dataset
    :param path: path to the dataset
    :return: train, valid, test
    """
    columns = ["head", "relation", "tail"]
    train = pd.read_csv(path + "/train.txt", sep="\t", header=None, names=columns)
    valid = pd.read_csv(path + "/valid.txt", sep="\t", header=None, names=columns)
    test = pd.read_csv(path + "/test.txt", sep="\t", header=None, names=columns)

    return train, valid, test


# Data visualization
def get_hist(data, bins=False):
    """
    Get histogram of the data using Freedman-Diaconis rule
    :param data: data to plot
    :param bins: number of bins or false to use Freedman-Diaconis rule
    :return: None
    """
    if bins:
        sns.displot(data, bins=bins, kde=True)
        return None
    
    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1 / 3)
    bins = round((data.max() - data.min()) / bin_width)
    print("Freedman-Diaconis number of bins:", bins)

    sns.displot(data, bins=bins, kde=True)

    return None
