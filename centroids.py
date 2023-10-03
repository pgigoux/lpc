import matplotlib.pyplot as plt
import numpy as np
from photutils.centroids import centroid_sources, centroid_2dg
from astropy.io import fits


def read_fits_image(file_name: str) -> np.ndarray:
    """
    Read the pixels from a FITS image
    :param file_name: FITS file name
    :return: ndarray with the pixel data
    """
    hdu_list = fits.open(file_name)
    hdu_list.info()
    return hdu_list[0].data


def sample_code(image_data: np.ndarray):
    """
    This is a piece of sample code largely based on the photutils.centroids documentation
    https://photutils.readthedocs.io/en/stable/centroids.html
    Used for initial testing
    :param image_data: image pixel data
    """

    # Calculate initial positions
    y_max, x_max = image_data.shape
    x1, y1 = x_max * 0.25, y_max * 0.25
    x2, y2 = x_max * 0.75, y1
    x3, y3 = x1, y_max * 0.75
    x4, y4 = x2, y3

    x_init = (x1, x2, x3, x4)
    y_init = (y1, y2, y3, y4)

    # Calculate the centroids
    x, y = centroid_sources(image_data, x_init, y_init, box_size=201,
                            centroid_func=centroid_2dg)

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.imshow(image_data, origin='lower', interpolation='nearest', cmap='gray')
    plt.scatter(x, y, marker='+', s=80, color='red', label='Centroids')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image = read_fits_image('test1.fits')
    sample_code(image)
