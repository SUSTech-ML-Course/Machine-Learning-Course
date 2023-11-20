import cv2
import numpy as np
from skimage import feature


class Descriptor:
    """
    Class that combines feature descriptors into a single descriptor
    to produce a feature vector for an input image.
    """

    class _skHOGDescriptor:

        """
        Wrapper subclass for skimage.feature.hog. Wrapping skimage.feature.hog
        in a class in which we also define a compute() function allows us to
        mirror the usage of OpenCV cv2.HOGDescriptor class method compute().
        """

        def __init__(self, hog_bins, pix_per_cell, cells_per_block,
                     block_norm, transform_sqrt):
            """@see Descriptor.#__init__(...)"""

            self.hog_bins = hog_bins
            self.pix_per_cell = pix_per_cell
            self.cells_per_block = cells_per_block
            self.block_norm = block_norm
            self.transform_sqrt = transform_sqrt

        def compute(self, image):
            multichannel = len(image.shape) > 2
            sk_hog_vector = feature.hog(image, orientations=self.hog_bins,
                                        pixels_per_cell=self.pix_per_cell,
                                        cells_per_block=self.cells_per_block,
                                        block_norm=self.block_norm, transform_sqrt=self.transform_sqrt,
                                        multichannel=multichannel, feature_vector=True)
            return np.expand_dims(sk_hog_vector, 1)

    def __init__(self, hog_features=False, hist_features=False,
                 spatial_features=False, hog_lib="cv", size=(64, 64),
                 hog_bins=9, pix_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_stride=None, block_norm="L1", transform_sqrt=True,
                 signed_gradient=False, hist_bins=16, spatial_size=(16, 16)):

        """
        Set feature parameters. For HOG features, either the OpenCV
        implementation (cv2.HOGDescriptor) or scikit-image implementation
        (skimage.feature.hog) may be selected via @param hog_lib. Some
        parameters apply to only one implementation (indicated below).

        @param hog_features (bool): Include HOG features in feature vector.
        @param hist_features (bool): Include color channel histogram features
            in feature vector.
        @param spatial_features (bool): Include spatial features in feature vector.
        @param size (int, int): Resize images to this (width, height) before
            computing features.
        @param hog_lib ["cv", "sk"]: Select the library to be used for HOG
            implementation. "cv" selects OpenCV (@see cv2.HOGDescriptor).
            "sk" selects scikit-image (@see skimage.feature.hog).
        @param pix_per_cell (int, int): HOG pixels per cell.
        @param cells_per_block (int, int): HOG cells per block.
        @param block_stride (int, int): [OpenCV only] Number of pixels by which
            to shift block during HOG block normalization. Defaults to half of
            cells_per_block.
        @param block_norm: [scikit-image only] Block normalization method for
            HOG. OpenCV uses L2-Hys.
        @param transform_sqrt (bool): [scikit-image only].
            @see skimage.feature.hog 
        @param hog_bins (int): Number of HOG gradient histogram bins.
        @param signed_gradient (bool): [OpenCV only] Use signed gradient (True)
            or unsigned gradient (False) for HOG. Currently, scikit-image HOG
            only supports unsigned gradients.
        @param hist_bins (int): Number of color histogram bins per color channel.
        @param spatial_size (int, int): Resize images to (width, height) for
            spatial binning.
        """

        self.hog_features = hog_features
        self.hist_features = hist_features
        self.spatial_features = spatial_features
        self.size = size
        self.hog_lib = hog_lib
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block

        if hog_lib == "cv":
            winSize = size
            cellSize = pix_per_cell
            blockSize = (cells_per_block[0] * cellSize[0],
                         cells_per_block[1] * cellSize[1])

            if block_stride is not None:
                blockStride = self.block_stride
            else:
                blockStride = (int(blockSize[0] / 2), int(blockSize[1] / 2))

            nbins = hog_bins
            derivAperture = 1
            winSigma = -1.
            histogramNormType = 0  # L2Hys (currently the only available option)
            L2HysThreshold = 0.2
            gammaCorrection = 1
            nlevels = 64
            signedGradients = signed_gradient

            self.HOGDescriptor = cv2.HOGDescriptor(winSize, blockSize,
                                                   blockStride, cellSize, nbins, derivAperture, winSigma,
                                                   histogramNormType, L2HysThreshold, gammaCorrection,
                                                   nlevels, signedGradients)
        else:
            self.HOGDescriptor = self._skHOGDescriptor(hog_bins, pix_per_cell,
                                                       cells_per_block, block_norm, transform_sqrt)

        self.hist_bins = hist_bins
        self.spatial_size = spatial_size

    def getFeatureVector(self, image):

        """Return the feature vector for an image."""

        if image.shape[:2] != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        feature_vector = np.array([])

        if self.hog_features:
            imagefeat = self.HOGDescriptor.compute(image)
            # feature_vector = imagefeat
            feature_vector = np.hstack(
                (feature_vector, imagefeat.flatten()))

        if self.hist_features:
            # np.histogram() returns a tuple if given a 2D array and an array
            # if given a 3D array. To maintain compatibility with other
            # functions in the object detection pipeline, check that the input
            # array has three dimensions. Add axis if necessary.
            # Note that histogram bin range assumes uint8 array.
            if len(image.shape) < 3:
                image = image[:, :, np.newaxis]

            hist_vector = np.array([])
            for channel in range(image.shape[2]):
                channel_hist = np.histogram(image[:, :, channel],
                                            bins=self.hist_bins, range=(0, 255))[0]
                hist_vector = np.hstack((hist_vector, channel_hist))
            feature_vector = np.hstack((feature_vector, hist_vector))

        if self.spatial_features:
            spatial_image = cv2.resize(image, self.spatial_size,
                                       interpolation=cv2.INTER_AREA)
            spatial_vector = spatial_image.ravel()
            feature_vector = np.hstack((feature_vector, spatial_vector))

        return feature_vector
