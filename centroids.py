"""
The code in this module is the basis of how the photometry code would work.
In the current implementation, the image data is read from an FITS file, but in
the LPC the image date will be available in an areaDetector array.
The conversion of the centroids from pixel to sky coordinates is also pending.
A distortion correction model might be needed.
"""
import matplotlib.pyplot as plt
import numpy as np
from photutils.centroids import centroid_sources, centroid_2dg
from astropy.io import fits


# def print_list(in_list: list):
#     for e in in_list:
#         print(e)


def read_fits_image(file_name: str) -> np.ndarray:
    """
    Read the pixels from a FITS image
    :param file_name: FITS file name
    :return: ndarray with the pixel data
    """
    hdu_list = fits.open(file_name)
    # hdu_list.info()
    return hdu_list[0].data


def get_reference_positions(image_data: np.ndarray) -> tuple:
    """
    Get the LGS reference positions
    For the time being we are assuming they are located at the center of each quadrant.
    In the final implementation they will most likely be read from a configuration file.
    The x,y reference coordinates will be grouped into two separate tuples.
    :return: x and y coordinates
    """
    y_max, x_max = image_data.shape
    x1, y1 = x_max * 0.25, y_max * 0.25
    x2, y2 = x_max * 0.75, y1
    x3, y3 = x1, y_max * 0.75
    x4, y4 = x2, y3
    return (x1, x2, x3, x4), (y1, y2, y3, y4)


def find_centroids(image_data: np.ndarray, reference_coordinates: tuple, box_size=101) -> list:
    """
    Find centroids in pixel space
    There results will be returned in a list of tuples, with each element containing the centroid
    x, y coordinate and the pixel value at that (approximate) location.
    :param image_data: image data
    :param reference_coordinates: tuple containing the x and y starting positions
    :param box_size: size of the sub-image used to find the centroid (must be an even number)
    :return: centroid list
    """
    # Extract the initial positions
    x_init, y_init = reference_coordinates

    # Make sure the box size is an odd number (required by centroid_sources)
    box_size = box_size if box_size % 2 == 1 else box_size + 1

    # Determine the centroid positions and pack the results
    xc, yc = centroid_sources(image_data, x_init, y_init, box_size=box_size,
                              centroid_func=centroid_2dg)
    centroid_list = [(x, y, image_data.item(int(y), int(x))) for x, y in zip(xc, yc)]
    # print_list(cent_list)

    return centroid_list


def plot_results(image_data: np.ndarray, centroid_list: list, min_counts=0):
    """
    :param image_data: image data
    :param centroid_list: centroid position and pixel values
    :param min_counts: minimum centroid counts to be valid
    """
    # Extract centroid coordinates that are above the minimum threshold
    xc, yc = [], []
    for c in centroid_list:
        x, y, v = c
        if v > min_counts:
            xc.append(x)
            yc.append(y)

    # Plot a mark over the valid centroids
    plt.figure(figsize=(8, 4))
    plt.imshow(image_data, origin='lower', interpolation='nearest', cmap='gray')
    plt.scatter(xc, yc, marker='+', s=80, color='red', label='Centroids')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image = read_fits_image('test2.fits')
    ref_pos = get_reference_positions(image)
    centroids = find_centroids(image, ref_pos, box_size=200)
    plot_results(image, centroids, min_counts=1500)
