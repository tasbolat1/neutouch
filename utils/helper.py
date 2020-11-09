import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# define GLOBAL VARIABLES
ASSETS_DIR = 'assets/'

# upload the taxel locations
left_taxels = np.load(ASSETS_DIR + 'left_tax_locs.npy') # left
right_taxels = np.load(ASSETS_DIR + 'right_tax_locs.npy') # right
empty_taxels = right_taxels.copy() # make empty taxels to help with plotting
empty_taxels[right_taxels!=0] = 0 # 0 means no value, 255 means full

left_finger_idx = np.unique(left_taxels)[1:]
right_taxels_idx = np.unique(right_taxels)[1:]

def plot_taxels(x, ax, finger='left'):
    """plots taxel intensity on taxel locations

    Input: x is a 2xN, where N is number of points.
    First column represent taxel locations, second one is for 
    taxel intensitites
    """
    if finger=='left':
        finger_idx = left_finger_idx
        taxels = left_taxels
    else:
        finger_idx = right_taxels_idx
        taxels = right_taxels
    for i, j in x:
        if i in left_finger_idx:
            empty_taxels[taxels==i] = j
    ax.imshow(empty_taxels, cmap='Blues')


def read_tac_file(fname):
	"""returns dataframe of tac file

	this is current version
	read .tac formatted file
	"""
	df = pd.read_csv(fname, names=['isNeg', 'taxel', 'time', 'read_time', 'parse_time'], sep=' ')
	#df.time = df.time - df.time[0]
	return df.drop(['read_time', 'parse_time'], axis=1)


def read_tac_file2(fname, start_time):
    """returns dataframe of tac file
    start_time must be provided
    this is current version
    read .tac formatted file
    """
    df = pd.read_csv(fname, names=['isNeg', 'taxel', 'time', 'read_time', 'parse_time'], sep=' ')
    df.time = start_time - df.time
    return df.drop(['read_time', 'parse_time'], axis=1)

def read_robotiq(fname):
    """returns dataframe of rbtq file and start time

    this is current version
    read .rbtq formatted file
    """
    df = pd.read_csv(fname, names=['time_s', 'time_ns', 'c1', 'c2', 'c3', 'c4', 'c5', 'target', 'current', 'c6'], sep=' ', index_col=False)
    df = df.assign(time = df.time_s+df.time_ns/1e9)
    return df[['time', 'target', 'current']], df.time.iloc[-1]