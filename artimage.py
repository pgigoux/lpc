"""
Code based on the Astropy documentation
1.2. Construction of an artificial (but realistic) image
https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-03-Construction-of-an-artificial-but-realistic-image.html
"""

import argparse
import random
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import QTable
from astropy.io import fits
from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image

# Image size
DEFAULT_NAXIS1 = 1760
DEFAULT_NAXIS2 = 2328

# Parameters used to define the field stars
DEFAULT_NGS_STARS = 40
DEFAULT_NGS_AMPLITUDE = 10000  # counts
DEFAULT_NGS_STDDEV = 4

# Parameters used to define the lgs stars
DEFAULT_LGS_ENABLE = [True, True, True, True]
DEFAULT_LGS_AMPLITUDE = 12000  # counts
DEFAULT_LGS_STDDEV = 10
DEFAULT_NOISE_PERCENT = 4
DEFAULT_LGS_LIST = [1, 2, 3, 4]

# Other parameters used to generate the artificial image
DEFAULT_SKY_BACKGROUND = 20
DEFAULT_BIAS_LEVEL = 1100
DEFAULT_DARK_CURRENT = 10  # electrons/pixel/second
DEFAULT_READOUT_NOISE = 10  # electrons

# This is the generator to use for any image component which changes in each image, e.g. read noise
# or Poisson error
noise_rng = np.random.default_rng(random.randint(1000, 5000))


def random_noise(max_value: int, percent=DEFAULT_NOISE_PERCENT) -> float:
    """
    Generate random noise as a percent of the max value.
    Used to introduce randomness in the lgs star poistions, amplitudes, etc.
    :param max_value: max value
    :param percent: noise percent
    :return: random noise
    """
    return (random.random() - 0.5) * max_value * percent / 100.0


def print_statistics(label: str, image: np.ndarray):
    """
    Print image statistics
    :param label: line label
    :param image: image
    """
    print(f'{label}: min={image.min(initial=0)}, max={image.max(initial=0)}, mean={image.mean()} std={image.std()}')


def read_noise(image, amount, gain=1) -> np.ndarray:
    """
    :param image: image that the noise array should match
    :param amount: amount of read noise (electrons)
    :param gain: camera gain (electrons/adu)
    :return: noise image
    """
    shape = image.shape
    noise = noise_rng.normal(scale=amount / gain, size=shape)
    return noise


def bias(image: np.ndarray, value: float, realistic=False) -> np.ndarray:
    """
    :param image: image that the noise array should match
    :param value: bias level to add
    :param realistic: add some columns with somewhat higher bias value?
    :return: bias image
    """
    # This is the whole thing: the bias is really suppose to be a constant offset!
    bias_im = np.zeros_like(image) + value

    # If we want a more realistic bias we need to do a little more work.
    if realistic:
        shape = image.shape
        number_of_colums = 5

        # We want a random-looking variation in the bias, but unlike the readout noise the bias should
        # *not* change from image to image, so we make sure to always generate the same "random" numbers.
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])

        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + col_pattern

    return bias_im


def dark_current(image: np.ndarray, current: float, exposure_time: float, gain=1.0, hot_pixels=False) -> np.ndarray:
    """
    Simulate dark current in a CCD, optionally including hot pixels.
    :param image: image that the noise array should match
    :param current: current value (electrons/pixel/second)
    :param exposure_time: exposure time (seconds)
    :param gain: camera gain (electrons/adu)
    :param hot_pixels: add hot pixels?
    :return: dark current image
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


def sky_background(image: np.ndarray, sky_counts: float, gain=1) -> np.ndarray:
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


def stars(image: np.ndarray, number, max_counts=DEFAULT_NGS_AMPLITUDE) -> np.ndarray:
    """
    Add stars to the image
    :param image: image that the noise array should match
    :param number: number of stars
    :param max_counts: maximum number of counts
    :return: image field with artificial stars
    """
    flux_range = [max_counts / 10, max_counts]

    y_max, x_max = image.shape
    x_mean_range = [0.1 * x_max, 0.9 * x_max]
    y_mean_range = [0.1 * y_max, 0.9 * y_max]
    x_stddev_range = [4, 4]
    y_stddev_range = [4, 4]
    params = dict([('amplitude', flux_range),
                   ('x_mean', x_mean_range),
                   ('y_mean', y_mean_range),
                   ('x_stddev', x_stddev_range),
                   ('y_stddev', y_stddev_range),
                   ('theta', [0, 2 * np.pi])])

    sources = make_random_gaussians_table(number, params, seed=12345)
    star_im = make_gaussian_sources_image(image.shape, sources)

    return star_im


def lgs_stars(image: np.ndarray, lgs_enable_factors: list,
              max_counts=DEFAULT_LGS_AMPLITUDE, stddev=DEFAULT_LGS_STDDEV) -> np.ndarray:
    """
    Create image with four lase guide stars centered on each quadrant
    The program assumes that there are at most four lgs centered on each quadrant.
    :param image: image that the noise array should match
    :param max_counts: maximum number of counts
    :param stddev: standard deviation around the star center
    :param lgs_enable_factors: list of multiplicative (0|1) factors used to disable lgs stars
    :return: image with lgs stars
    """
    shape = image.shape

    # Calculate the image positions
    y_max, x_max = shape

    x1 = x_max * 0.25 + random_noise(x_max)
    y1 = y_max * 0.25 + random_noise(y_max)
    x2 = x_max * 0.75 + random_noise(y_max)
    y2 = y1 + random_noise(y_max)
    x3 = x1 + random_noise(x_max)
    y3 = y_max * 0.75 + random_noise(y_max)
    x4 = x2 + random_noise(x_max)
    y4 = y3 + random_noise(y_max)

    x_stddev = [stddev + random_noise(stddev) for _ in range(4)]
    y_stddev = [stddev + random_noise(stddev) for _ in range(4)]

    # Calculate the maximum number of counts per star
    # The amplitudes will be zero of the particular lgs is disabled.
    amplitudes = [(max_counts + random_noise(max_counts)) * factor for factor in lgs_enable_factors]

    table = QTable()
    table['amplitude'] = amplitudes
    table['x_mean'] = [x1, x2, x3, x4]
    table['y_mean'] = [y1, y2, y3, y4]
    table['x_stddev'] = x_stddev
    table['y_stddev'] = y_stddev
    table['theta'] = np.radians(np.array([0, 0, 0, 0]))

    return make_gaussian_sources_image(shape, table)


def generate_artificial_image(lgs_enable_factors: list, statistics: Optional[bool] = False) -> np.ndarray:
    """
    :return: artificial image
    """
    base_image = np.zeros([DEFAULT_NAXIS1, DEFAULT_NAXIS2])
    if statistics:
        print_statistics('base', base_image)

    noise_image = read_noise(base_image, DEFAULT_READOUT_NOISE, gain=1)
    if statistics:
        print_statistics('noise', noise_image)

    bias_image = bias(base_image, DEFAULT_BIAS_LEVEL, realistic=True)
    if statistics:
        print_statistics('bias', bias_image)

    sky_image = sky_background(base_image, DEFAULT_SKY_BACKGROUND)
    if statistics:
        print_statistics('sky', sky_image)

    star_image = stars(base_image, DEFAULT_NGS_STARS, max_counts=DEFAULT_NGS_AMPLITUDE)
    if statistics:
        print_statistics('star', star_image)

    lgs_image = lgs_stars(base_image, lgs_enable_factors, max_counts=DEFAULT_LGS_AMPLITUDE)
    if statistics:
        print_statistics('lgs', lgs_image)

    final_image = base_image + noise_image + bias_image + sky_image + star_image + lgs_image
    if statistics:
        print_statistics('final', final_image)

    return final_image


def write_fits_image(file_name: str, image: np.ndarray, statistics: Optional[bool] = False):
    """
    Create a FITS image on disk.
    The pixel data is forced to be 16-bit.
    The image header will contain random information since only a few fields are needed
    :param file_name: output image name
    :param image: image pixels
    :param statistics: print image statistics
    """
    # Create the image
    hdu = fits.PrimaryHDU(data=np.int16(image))
    if statistics:
        print_statistics('fits', hdu.data)

    # Create a standard image header
    # hdu.header['SIMPLE'] = True
    hdu.header['BITPIX'] = 16  # 16 bits per pixel
    hdu.header['NAXIS'] = 2
    hdu.header['NAXIS1'] = DEFAULT_NAXIS1
    hdu.header['NAXIS2'] = DEFAULT_NAXIS2
    hdu.header['BZERO'] = 0
    hdu.header['BSCALE'] = 1
    hdu.header['EXTEND'] = True
    hdu.header['GAIN'] = 200
    hdu.header['OFFSET'] = 100
    hdu.header['BIN_X'] = 2
    hdu.header['BIN_Y'] = 2
    hdu.header['RA'] = 26.554
    hdu.header['DEC'] = 89.842
    hdu.header['ALT'] = 90.
    hdu.header['AZ'] = 0.
    hdu.header['PAR'] = 0.
    hdu.header['CCD_TEMP'] = 26.5

    # Write the file to disk
    try:
        hdu.writeto(file_name, overwrite=True)

    except Exception as e:
        print(f'Cannot write output file {file_name} - {e}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate an artificial LPC image')

    parser.add_argument(action='store',
                        dest='filename',
                        help='output file name')

    parser.add_argument('-p', '--plot',
                        action='store_true',
                        dest='plot',
                        default=False,
                        help='plot image')

    parser.add_argument('-s', '--stat',
                        action='store_true',
                        dest='statistics',
                        default=False,
                        help='print image statistics')

    parser.add_argument('-e', '--enable',
                        nargs='+',
                        action='store',
                        dest='enable',
                        default=[],
                        help='enable individual lgs (1,2,3,4 [default=all]')

    # args = parser.parse_args(['test.fits', '-p', '-e', '1', '4'])
    args = parser.parse_args(['test.fits', '-p', '-s'])
    # args = parser.parse_args()

    # Convert enable flags to multiplicative factors

    if args.enable:
        lgs_enable = [0 for _ in range(4)]
        for n in args.enable:
            lgs_enable[int(n) - 1] = 1
    else:
        lgs_enable = [1 for _ in range(4)]

    # Generate and save image
    art_image = generate_artificial_image(lgs_enable, args.statistics)
    write_fits_image('test.fits', art_image, statistics=args.statistics)

    # Plot image
    if args.plot:
        plt.imshow(art_image, cmap='gray')
        plt.show()
