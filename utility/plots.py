import matplotlib
if __name__ != "__main__":
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(data, height, width, dpi=100.0, rgbonly=False):
    n = data.shape[1]
    x = np.linspace(0, 1, n)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.grid()
    styles = ["r-", "g-", "b-", "c-", "m-", "y-", "k-"]
    if rgbonly:
        styles = styles[:3]
    for y, style in zip(data, styles):
        plt.plot(x, y, style, lw=2)
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tick_params(axis='both', which='both', labelbottom=False,
                    labelleft=False)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    bin = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    plt.close()
    return bin.reshape([h, w, 3])


def make_test_image(samples, batch_size):
    a = np.linspace(0, 1, samples, dtype=np.float32)
    z = np.zeros_like(a)
    im = np.stack([a, z, z,
                   z, a, z,
                   z, z, a,
                   z, a, a,
                   a, z, a,
                   a, a, z,
                   a, a, a])
    im = im.reshape([7, 3, samples])
    im = im.transpose([1, 0, 2])
    im = np.tile(im[None, :], [batch_size, 1, 1, 1])
    return im


def plots_from_test_image(im, height, width, rgbonly=False):
    channels = [im[:, 0, 0, :],
                im[:, 1, 1, :],
                im[:, 2, 2, :]]
    channels.append((channels[1] + channels[2]) / 2)
    channels.append((channels[0] + channels[2]) / 2)
    channels.append((channels[0] + channels[1]) / 2)
    channels.append((channels[0] + channels[1] + channels[2]) / 3)
    im = np.stack(channels, 1)
    n = im.shape[0]
    pls = [make_plots(im[i, :], height, width, rgbonly=rgbonly) for i in range(n)]
    return np.stack(pls, 0).transpose([0, 3, 1, 2]) / 255.0


def _main():
    data = np.outer(np.arange(3, 10), np.arange(11)) / 60.0
    # data = np.random.rand(7, 10)
    im = make_plots(data, 256, 256)
    print(im.shape, im.dtype)
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    _main()
