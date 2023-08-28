"""
This script is used to generate the images for the fruit classifier.
The script will randomly select a background image and a fruit image.
Then, it will randomly resize the fruit image and place it on the background image.
The script will generate N variations per background image.
The generated images will be saved in the generated_images folder.

The resize factor is randomly selected between MIN_RESIZE_FACTOR and MAX_RESIZE_FACTOR.
"""

import cv2
import numpy as np
import glob
from copy import deepcopy

# Settings
VARIATION_PER_BACKGROUND = 10  # The number of variations per background
MAX_RESIZE_FACTOR = 1.4  # The maximum resize factor for the image, shrink or enlarge
MIN_RESIZE_FACTOR = 0.3  # The minimum resize factor for the image, shrink or enlarge


def randomise_image():
    """
    Randomise the image by applying a random transformation and background.

    @return: none
    """

    print('Randomising image...')

    # get the fruit images path
    fruit_images_path = glob.glob('fruit_images/*')

    # get the background images path
    background_images_path = glob.glob('background_images/*')

    total = len(background_images_path) * len(fruit_images_path) * VARIATION_PER_BACKGROUND

    count = 0
    for background in background_images_path:
        for fruit_path in fruit_images_path:
            # read the fruit image
            fruit_image = cv2.imread(fruit_path)
            # read the background image
            background_image = cv2.imread(background)

            # resize the background image to 320 x 240 pixels if it is not
            if background_image.shape[0] != 240 or background_image.shape[1] != 320:
                background_image = cv2.resize(background_image, (320, 240))

            fruit = get_fruit_roi(fruit_image)

            for n in range(VARIATION_PER_BACKGROUND):
                # resize the fruit image
                fruit_w = fruit.shape[1]
                fruit_h = fruit.shape[0]

                scale = np.random.uniform(MIN_RESIZE_FACTOR, MAX_RESIZE_FACTOR)

                # resize the fruit image
                fruit_fresh = cv2.resize(fruit, (int(fruit_w * scale), int(fruit_h * scale)))

                # get the fruit image width and height
                fruit_w = fruit_fresh.shape[1]
                fruit_h = fruit_fresh.shape[0]

                # get the background image width and height
                background_w = background_image.shape[1]
                background_h = background_image.shape[0]

                # get the random x and y position of the fruit
                x = int(np.random.uniform(-fruit_w/2, background_w - fruit_w / 2))
                y = int(np.random.uniform(-fruit_h/2, background_h - fruit_h / 2))

                # crop the fruit image if it is out of the background
                if x < 0:
                    fruit_fresh = fruit_fresh[:, -x:]
                    fruit_w = fruit_fresh.shape[1]
                    x = 0

                if y < 0:
                    fruit_fresh = fruit_fresh[-y:, :]
                    fruit_h = fruit_fresh.shape[0]
                    y = 0

                if x + fruit_w > background_w:
                    fruit_fresh = fruit_fresh[:, :background_w - x]
                    fruit_w = fruit_fresh.shape[1]

                if y + fruit_h > background_h:
                    fruit_fresh = fruit_fresh[:background_h - y, :]
                    fruit_h = fruit_fresh.shape[0]

                # get the ROI of the background
                roi = background_image[y:y + fruit_h, x:x + fruit_w]

                # get the mask of the fruit
                fruit_gray = cv2.cvtColor(fruit_fresh, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(fruit_gray, 10, 255, cv2.THRESH_BINARY)

                # invert the mask
                mask_inv = cv2.bitwise_not(mask)

                # get the background of the fruit
                background_of_fruit = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # get the fruit
                fruit_of_fruit = cv2.bitwise_and(fruit_fresh, fruit_fresh, mask=mask)

                # add the fruit to the background
                dst = cv2.add(background_of_fruit, fruit_of_fruit)

                # add the fruit to the background image
                output_image = background_image.copy()
                output_image[y:y + fruit_h, x:x + fruit_w] = dst

                # save the image
                cv2.imwrite(f'generated_images/fruit_output_{count}.png', output_image)

                count += 1
                print('Progress: ' + str(count) + '/' + str(total) + ' (' + str(round(count / total * 100, 2)) + '%)')


def get_fruit_roi(image):
    """Get the region of interest of the fruit in the image.

    Cropping the empty space around the fruit.

    @param image: The image to get the ROI from.
    @return: The ROI of the fruit.
    """

    h = image.shape[0]
    w = image.shape[1]

    # initialise the left, right, top and bottom of the fruit
    left = -1
    right = -1
    top = -1
    bottom = -1

    # crop the empty space around the fruit
    for i in range(w):
        if left > 0 and right > 0:
            break

        if np.sum(image[:, i]) > 0 > left:
            left = i - 1

        if np.sum(image[:, w - i - 1]) > 0 > right:
            right = w - i + 1

    for i in range(h):
        if top > 0 and bottom > 0:
            break

        if np.sum(image[i, :]) > 0 > top:
            top = i - 1

        if np.sum(image[h - i - 1, :]) > 0 > bottom:
            bottom = h - i + 1

    # crop the image
    return deepcopy(image[top:bottom, left:right])


if __name__ == '__main__':
    print('Randomising fruit images...')

    randomise_image()
