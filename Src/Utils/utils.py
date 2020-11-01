from __future__ import print_function
import numpy as np
import shutil
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync
import importlib
from time import time
import sys

np.random.seed(0)

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore, method):
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        self.log.flush()
        fsync(self.log.fileno())



def save_plots(rewards, config):
    np.save(config.paths['results'] + "rewards", rewards)
    if config.debug:
        if 'Grid' in config.env_name or 'room' in config.env_name:
            # Save the heatmap
            plt.figure()
            plt.title("Exploration Heatmap")
            plt.xlabel("100x position in x coordinate")
            plt.ylabel("100x position in y coordinate")
            plt.imshow(config.env.heatmap, cmap='hot', interpolation='nearest', origin='lower')
            plt.savefig(config.paths['results'] + 'heatmap.png')
            np.save(config.paths['results'] + "heatmap", config.env.heatmap)
            config.env.heatmap.fill(0)  # reset the heat map
            plt.close()

        plt.figure()
        plt.ylabel("Total return")
        plt.xlabel("Episode")
        plt.title("Performance")
        plt.plot(rewards)
        plt.savefig(config.paths['results'] + "performance.png")
        plt.close()


def plot(rewards):
    # Plot the results
    plt.figure(1)
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Trajectories")
    plt.ylabel("Reward")
    plt.title("Baseline Reward")
    plt.show()



class Linear_schedule:
    def __init__(self, max_len, max=1, min=0):
        self.max_len = max_len
        self.max = max
        self.min = min
        self.temp = self.max/self.max_len

    def get(self, idx):
        return max(self.min, (self.max_len - idx) * self.temp)

class Power_schedule:
    def __init__(self, pow, max=1, min=0):
        self.pow = pow
        self.min = min
        self.temp = max

    def get(self, idx=-1):
        self.temp *= self.pow
        return max(self.min, self.temp)


def binaryEncoding(num, size):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % 2
        num = num//2
        i -= 1
    return binary


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)



def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location

def dynamic_load(dir, name, load_class=False):
    try:
        abs_path = search(dir, name).split('/')[1:]
        pos = abs_path.index('SAS_RL')
        module_path = '.'.join([str(item) for item in abs_path[pos + 1:]])
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except:
        raise ValueError("Failed to dynamically load the class: " + name )

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    dir_path = str.split(dir_path, sep='/')[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join('/', *(dir_path[:i + 1])))


def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def clip_norm(params, max_norm=1):
    # return params
    norm_param = []
    for param in params:
        norm = np.linalg.norm(param, 2)
        if norm > max_norm:
            norm_param.append(param/norm * max_norm)
        else:
            norm_param.append(param)
    return norm_param


