import matplotlib.pyplot as plt
import numpy as np
from photutils.centroids import centroid_com, centroid_sources
from photutils.datasets import make_4gaussians_image
from astropy.io import fits


def open_fits_image(file_name: str) -> np.ndarray:
    hdu_list = fits.open(file_name)
    hdu_list.info()
    return hdu_list[0].data


def something(data: np.ndarray):
    # data = make_4gaussians_image()

    y_max, x_max = data.shape
    print(x_max, y_max)
    print(data.mean())

    x1 = x_max * 0.25
    y1 = y_max * 0.25
    x2 = x_max * 0.75
    y2 = y1
    x3 = x1
    y3 = y_max * 0.75
    x4 = x2
    y4 = y3

    x_init = (x1, x2, x3, x4)
    y_init = (y1, y2, y3, y4)

    print(x_init, y_init)
    x, y = centroid_sources(data, x_init, y_init, box_size=101,
                            centroid_func=centroid_com)

    # x, y = centroid_com(data[0:int(x_max/2), 0:int(y_max/2)])
    print(x, y)

    plt.figure(figsize=(8, 4))
    plt.imshow(data, origin='lower', interpolation='nearest', cmap='gray')
    plt.scatter(x_init, y_init, marker='+', s=80, color='red', label='Centroids')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image = open_fits_image('test.fits')
    something(image)


