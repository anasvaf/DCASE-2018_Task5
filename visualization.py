from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_pca(n_classes, z, labels, save_path):
    if z.shape[0] == 2:
        plt.figure()
        plt.scatter(z[:, 0], z[:, 1],
                    c=np.argmax(labels, 1),
                    marker='o',
                    edgecolor='none',
                    cmap=discrete_cmap(n_classes, 'jet'),
                    s=8)
        plt.colorbar(ticks=range(n_classes))
        axes = plt.gca()
        plt.grid(True)
        plt.savefig(save_path)
    else:
        pca = PCA(n_components=2)
        z_transformed = pca.fit(z).transform(z)
        plt.figure()
        plt.scatter(z[:, 0], z[:, 1],
                    c=labels,
                    marker='o',
                    edgecolor='none',
                    cmap=discrete_cmap(n_classes, 'jet'),
                    s=8)
        plt.colorbar(ticks=range(n_classes))
        axes = plt.gca()
        plt.grid(True)
        plt.savefig(save_path)


def show_image(img_vec, save_path):
    fig = plt.figure()
    plt.imshow(img_vec, cmap='gray')
    plt.savefig(save_path)
    plt.close(fig)
