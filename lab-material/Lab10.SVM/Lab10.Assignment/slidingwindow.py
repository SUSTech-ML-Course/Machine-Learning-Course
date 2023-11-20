import numpy as np
import matplotlib.pyplot as plt
import cv2


def slidingWindow(image_size, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                  x_range=(0, 1), y_range=(0, 1), scale=1.5):
    """
    Run a sliding window across an input image and return a list of the
    coordinates of each window.

    Window travels the width of the image (in the +x direction) at a range of
    heights (toward the bottom of the image in the +y direction). At each
    successive y, the size of the window is increased by a factor equal to
    @param scale. The horizontal search area is limited by @param x_range
    and the vertical search area by @param y_range.

    @param image_size (int, int): Size of the image (width, height) in pixels.
    @param init_size (int, int): Initial size of of the window (width, height)
        in pixels at the initial y, given by @param y_range[0].
    @param x_overlap (float): Overlap between adjacent windows at a given y
        as a float in the interval [0, 1), where 0 represents no overlap
        and 1 represents 100% overlap.
    @param y_step (float): Distance between successive heights y as a
        fraction between (0, 1) of the total height of the image.
    @param x_range (float, float): (min, max) bounds of the horizontal search
        area as a fraction of the total width of the image.
    @param y_range (float, float) (min, max) bounds of the vertical search
        area as a fraction of the total height of the image.
    @param scale (float): Factor by which to scale up window size at each y.
    @return windows: List of tuples, where each tuple represents the
        coordinates of a window in the following order: (upper left corner
        x coord, upper left corner y coord, lower right corner x coord,
        lower right corner y coord).
    """

    windows = []
    h, w = image_size[1], image_size[0]
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * h)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        if y + win_height > int(y_range[1] * h) or win_width > w:
            break
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            windows.append((x, y, x + win_width, y + win_height))

    return windows


def display_windows(img: str, color=(0, 0, 255), thick=6):
    """
    Shows all windows of slidingWindow() in an image

    @param img:     path of img you want to show
    @param color:   windows' color drawn on img
    @param thick:   width of windows' edges
    @return         an img on which windows are drawn
    """
    image = plt.imread(img)
    h, w, c = image.shape
    windows = slidingWindow((w, h))
    rects = []
    for w in windows:
        rects.append(((int(w[0]), int(w[1])), (int(w[2]), int(w[3]))))

    random_color = False
    # Iterate through windows
    for rect in rects:
        if color == 'random' or random_color:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            random_color = True
        # Draw a rectangle given windows coordinates
        cv2.rectangle(image, rect[0], rect[1], color, thick)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

    return image
