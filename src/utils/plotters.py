import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    def plot_data(self, x, y, title, xlabel='', ylabel='', size=(5,5), label=None, save_path=None):
        fig = plt.figure(figsize=size)
        plt.plot(x, y, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            file_path = os.path.join(save_path, f'{title}.png')
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

    def plot_statistics(self, recorder, save_path):
        statistics = recorder.report_statistics()
        x = statistics['n_episode']
        for k, v in statistics.items():
            if k == 'n_episode':
                continue

            self.plot_data(x, v, k, 'n_episode', size=(5,5), save_path=save_path)