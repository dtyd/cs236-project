import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch
import pdb


if __name__ == "__main__":
    pdb.set_trace()
    gen_imgs = np.load('samples.out.npy')
    gen_imgs = np.clip(gen_imgs, 0, 1)
    captions = ['cat', 'bird', 'truck', 'horse']
    for i in range(gen_imgs.shape[0]):
        plt.imshow(gen_imgs[i].transpose([1, 2, 0]))
        plt.title(captions[i])
        plt.savefig('sample_' + str(i) + '.png')
        plt.show()
    plt.close()


def load_model(file_path, generative_model):
    dict = torch.load(file_path, map_location='cpu')
    generative_model.load_state_dict(dict)