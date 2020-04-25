#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import scipy.stats as stats
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# This module loads the images into NumPy arrays. 
def get_img_array(imgs_dir):
    return [get_array(img) for img in os.listdir(imgs_dir)]

# This module gets the height and width values for the padding. 
def get_h_w(img_arrs):
    return max([[img_arrs[i].shape[0], img_arrs[i].shape[1]] for i in range(len(img_arrs))])

# Adding the padding
def get_padded(img_arrs, h, w):
    return [pad(img_arrs[i], h, w) for i in range(len(img_arrs))]

# Normanizing the images. 
def normalize(x):
    return (x - 128.0) / 128
# This module loads the images into NumPy arrays
def get_array(img):
    im = Image.open(img)
    return np.array(im)

# For histogram plotting 
def plot_hist(img, col): 
    plt.hist(img, color = col, alpha = 0.5)
    plt.yscale('log')

# Get the Grayscale image. 
def rgb2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Plot the distributions
def get_dist(channel, img):
    if channel == 'gray':
        return rgb2gray(img).ravel()
    colors = ['red', 'green', 'blue']
    for i in range(len(colors)): 
        if channel == colors[i]:
            return img[:, :, i].ravel()

# Get the values in the Pandas DataFrames. 
def get_df(channel, imgs_dir, padded_imgs):
    names_arr = ['Img index', 'Img name', ' Mean', ' Median', ' Mode']
    names_arr[2:] = [channel + names_arr[i] for i in range(2, len(names_arr))]
    df = pd.DataFrame(columns=names_arr)
    for i in range(len(padded_imgs)):
        dist = get_dist(channel, padded_imgs[i])
        df = df.append({'Img index': i, 'Img name': os.listdir(imgs_dir)[i], names_arr[2]: np.mean(dist), names_arr[3]:np.median(dist),  names_arr[4]: stats.mode(dist)[0][0]}, ignore_index=True)
    return df

def get_histograms(channel, img, show):
    dist = get_dist(channel, img)
    plot_hist(dist, col=channel)
    if show: 
        plt.show()
        
# Get the triangle correlation map.         
def get_triangle_corr(corr):
    f, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(150, 275, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, alpha = 0.5)
    

