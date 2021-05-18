import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import os


def plot_results(data: dict, save_path: str, title: str):
    """Plot results histogram using matplotlib"""
    sns.set()
    for key in data.keys():
        data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.legend()
    plt.xlabel('Query times')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'output.jpg'))
    plt.show()

    sns.set()
    for key in data.keys():
        data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.legend()
    plt.xlabel('Query times')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1)
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'output_fine.jpg'))
    plt.show()

    sns.set()
    for key in data.keys():
        data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.legend()
    plt.xlabel('Query times')
    plt.ylabel('Accuracy')
    plt.xlim(0, 10)
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'output_start.jpg'))
    plt.show()



if __name__ == '__main__':
    log_path = 'MNIST_result/'
    results = {}
    npy_list =[
        'bald-MC_dropout=True.npy',
        'max_entropy-MC_dropout=True.npy',
        'min_entropy-MC_dropout=True.npy',
        'mean_std-MC_dropout=True.npy',
        'uniform-MC_dropout=True.npy',
        'var_ratios-MC_dropout=True.npy'
               ]
    for npy in npy_list:
        value = np.load(log_path+npy)
        key = npy.split('.')[0]
        results[key] = value

    plot_results(results, save_path=log_path, title='MNIST')