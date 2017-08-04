import numpy as np
rom matplotlib import pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_cam, overlay


def plot_heatmap(arr, figsize=(15, 15)):
    n = arr.shape[0]
    fig, ax = plt.subplots(1, n, figsize=figsize)
    ax = ax.ravel()
    for i in range(n):
        ax[i].imshow(arr[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])


def attention_dia(idx):
    pre_img = x_train[idx]
    filter_idx = np.argmax(my_model.predict(pre_img[np.newaxis]))
    heatmap = visualize_cam(my_model, layer_idx, filter_indices=filter_idx, seed_input=pre_img, backprop_modifier='guided')
    
    return overlay(churches[idx], heatmap)


num_idx = 5
layer_idx = utils.find_layer_idx(my_model, 'predictions')
dia_idx = random.sample(range(100), 10)
dia_attention_maps = np.array(map(attention_dia, dia_idx))

plot_heatmap(dia_attention_maps)