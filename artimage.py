"""
Code based on the Astropy documentation
1.2. Construction of an artificial (but realistic) image
https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-03-Construction-of-an-artificial-but-realistic-image.html
"""

import random
import numpy as np

from matplotlib import pyplot as plt
from astropy.table import QTable
from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image

# This is the generator to use for any image component which changes in each image, e.g. read noise
# or Poisson error
noise_rng = np.random.default_rng(random.randint(1000, 5000))


def read_noise(image, amount, gain=1) -> np.ndarray:
    """
    :param image: image that the noise array should match
    :param amount: ammount of read noise (electrons)
    :param gain: camera gain (electrons/adu)
    :return:
    """
    shape = image.shape
    noise = noise_rng.normal(scale=amount / gain, size=shape)
    return noise


def bias(image: np.ndarray, value, realistic=False) -> np.ndarray:
    """
    :param image: image that the noise array should match
    :param value: bias level to add
    :param realistic: add some columns with somewhat higher bias value?
    :return:
    """
    # This is the whole thing: the bias is really suppose to be a constant offset!
    bias_im = np.zeros_like(image) + value

    # If we want a more realistic bias we need to do a little more work.
    if realistic:
        shape = image.shape
        number_of_colums = 5

        # We want a random-looking variation in the bias, but unlike the readnoise the bias should
        # *not* change from image to image, so we make sure to always generate the same "random" numbers.
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])

        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + col_pattern

    return bias_im


def dark_current(image: np.ndarray, current, exposure_time, gain=1.0, hot_pixels=False) -> np.ndarray:
    """
    Simulate dark current in a CCD, optionally including hot pixels.
    :param image: image that the noise array should match
    :param current: current value (electrons/pixel/second)
    :param exposure_time: exposure time (seconds)
    :param gain: camera gain (electrons/adu)
    :param hot_pixels: add hot pixels?
    :return:
    """

    # dark current for every pixel; we'll modify the current for some pixels if
    # the user wants hot pixels.
    base_current = current * exposure_time / gain

    # This random number generation should change on each call.
    dark_im = noise_rng.poisson(base_current, size=image.shape)

    if hot_pixels:
        # We'll set 0.01% of the pixels to be hot; that is probably too high but should
        # ensure they are visible.
        y_max, x_max = dark_im.shape

        n_hot = int(0.0001 * x_max * y_max)

        # Like with the bias image, we want the hot pixels to always be in the same places
        # (at least for the same image size) but also want them to appear to be randomly
        # distributed. So we set a random number seed to ensure we always get the same thing.
        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)

        hot_current = 10000 * current

        dark_im[(hot_y, hot_x)] = hot_current * exposure_time / gain

    return dark_im


def sky_background(image: np.ndarray, sky_counts, gain=1) -> np.ndarray:
    """
    Generate sky background, optionally including a gradient across the image (because
    some times Moons happen).
    :param image: image that the noise array should match
    :param sky_counts: current value (electrons/pixel/second)
    :param gain: camera gain (electrons/adu)
    :return: dark image
    """
    sky_im = noise_rng.poisson(sky_counts * gain, size=image.shape) / gain
    return sky_im


def stars(image: np.ndarray, number, max_counts=10000) -> np.ndarray:
    """
    Add stars to the image
    :param image: image that the noise array should match
    :param number: number of stars
    :param max_counts:
    :return:
    """
    """
    Add some stars to the image.
    """
    flux_range = [max_counts / 10, max_counts]

    y_max, x_max = image.shape
    xmean_range = [0.1 * x_max, 0.9 * x_max]
    ymean_range = [0.1 * y_max, 0.9 * y_max]
    x_stddev_range = [4, 4]
    y_stddev_range = [4, 4]
    params = dict([('amplitude', flux_range),
                   ('x_mean', xmean_range),
                   ('y_mean', ymean_range),
                   ('x_stddev', x_stddev_range),
                   ('y_stddev', y_stddev_range),
                   ('theta', [0, 2 * np.pi])])

    sources = make_random_gaussians_table(number, params, seed=12345)
    star_im = make_gaussian_sources_image(image.shape, sources)

    return star_im


def noise(max_value: int, percent=5) -> float:
    """
    Generate random noise
    :param max_value: max value
    :param percent: noise percent
    :return:
    """
    return (random.random() - 0.5) * max_value * percent / 100.0


def lgs_stars(image: np.ndarray, amplitude=100) -> np.ndarray:
    """
    Create image with four lase guide stars centered on each quadrant
    :param image: image that the noise array should match
    :return:
    """
    shape = image.shape
    y_max, x_max = shape
    x_max_2 = x_max / 2
    y_max_2 = y_max / 2
    x1 = x_max / 4 + noise(x_max)
    y1 = y_max / 4 + noise(y_max)
    x2 = x1 + x_max_2 + noise(x_max)
    y2 = y1 + noise(y_max)
    x3 = x1 + noise(x_max)
    y3 = y1 + y_max_2 + noise(y_max)
    x4 = x2 + noise(x_max)
    y4 = y3 + noise(y_max)

    mean_stddev = 15
    x_stddev = [mean_stddev + noise(mean_stddev) for x in range(4)]
    y_stddev = [mean_stddev + noise(mean_stddev) for x in range(4)]

    amplitudes = [amplitude + noise(amplitude) for x in range(4)]

    table = QTable()
    table['amplitude'] = amplitudes
    table['x_mean'] = [x1, x2, x3, x4]
    table['y_mean'] = [y1, y2, y3, y4]
    table['x_stddev'] = x_stddev
    table['y_stddev'] = y_stddev
    # table['flux'] = [200, 200, 200, 200]
    table['theta'] = np.radians(np.array([100,100,100,100]))

    return make_gaussian_sources_image(shape, table)


if __name__ == '__main__':
    base_image = np.zeros([1000, 1000])
    noise_image = read_noise(base_image, 10, gain=1)
    bias_image = bias(base_image, 1100, realistic=True)
    sky_image = sky_background(base_image, 20)
    star_image = stars(base_image, 20)
    lgs_image = lgs_stars(base_image, amplitude=10000)

    final_image = base_image + noise_image + bias_image + sky_image + star_image + lgs_image

    plt.imshow(final_image, cmap='gray')
    plt.show()
