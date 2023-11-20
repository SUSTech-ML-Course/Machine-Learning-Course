from collections import deque
from datetime import datetime
import os
import pickle
import time
import cv2
import numpy as np
import tqdm
from scipy.ndimage.measurements import label
from descriptor import Descriptor
from slidingwindow import slidingWindow


class Detector:
    """
    Class for finding objects in a video stream. Loads and utilizes a
    pretrained classifier.
    """

    def __init__(self, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                 x_range=(0, 1), y_range=(0, 1), scale=1.5):

        """For input arguments, @see slidingwindow.#slidingWindow(...)"""

        self.init_size = init_size
        self.x_overlap = x_overlap
        self.y_step = y_step
        self.x_range = x_range
        self.y_range = y_range
        self.scale = scale
        self.windows = None

    def loadClassifier(self, filepath=None, classifier_data=None):

        """
        Load a classifier trained by the functions in train.py. Either a dict
        (classifier_data) or pickled file (filepath) may be supplied.
        """

        if filepath is not None:
            filepath = os.path.abspath(filepath)
            if not os.path.isfile(filepath):
                raise FileNotFoundError("File " + filepath + " does not exist.")
            classifier_data = pickle.load(open(filepath, "rb"))
        else:
            classifier_data = classifier_data

        if classifier_data is None:
            raise ValueError("Invalid classifier data supplied.")

        self.classifier = classifier_data["classifier"]
        self.scaler = classifier_data["scaler"]
        self.cv_color_const = classifier_data["cv_color_const"]
        self.channels = classifier_data["channels"]

        # Simply loading the descriptor from the dict with
        #   self.descriptor = classifier_data["descriptor"]
        # produces an error. Thus, we instantiate a new descriptor object
        # using the same parameters on which the classifier was trained.
        self.descriptor = Descriptor(
            hog_features=classifier_data["hog_features"],
            hist_features=classifier_data["hist_features"],
            spatial_features=classifier_data["spatial_features"],
            hog_lib=classifier_data["hog_lib"],
            size=classifier_data["size"],
            hog_bins=classifier_data["hog_bins"],
            pix_per_cell=classifier_data["pix_per_cell"],
            cells_per_block=classifier_data["cells_per_block"],
            block_stride=classifier_data["block_stride"],
            block_norm=classifier_data["block_norm"],
            transform_sqrt=classifier_data["transform_sqrt"],
            signed_gradient=classifier_data["signed_gradient"],
            hist_bins=classifier_data["hist_bins"],
            spatial_size=classifier_data["spatial_size"])

        return self

    def classify(self, image):

        """
        Classify windows of an image as "positive" (containing the desired
        object) or "negative". Return a list of positively classified windows.
        """

        if self.cv_color_const > -1:
            image = cv2.cvtColor(image, self.cv_color_const)

        if len(image.shape) > 2:
            image = image[:, :, self.channels]
        else:
            image = image[:, :, np.newaxis]

        feature_vectors = [self.descriptor.getFeatureVector(
            image[y_upper:y_lower, x_upper:x_lower, :])
            for (x_upper, y_upper, x_lower, y_lower) in self.windows]

        # Scale feature vectors, predict, and return predictions.
        feature_vectors = self.scaler.transform(feature_vectors)
        predictions = self.classifier.predict(feature_vectors)
        return [self.windows[ind] for ind in np.argwhere(predictions == 1)[:, 0]]

    def detectVideo(self, video_capture=None, num_frames=9, threshold=120,
                    min_bbox=None, show_video=True, draw_heatmap=True,
                    draw_heatmap_size=0.2, write=True, write_fps=24):

        """
        Find objects in each frame of a video stream by integrating bounding
        boxes over several frames to produce a heatmap of pixels with high
        prediction density, ignoring pixels below a threshold, and grouping
        the remaining pixels into objects. Draw boxes around detected objects.

        @param video_capture (VideoCapture): cv2.VideoCapture object.
        @param num_frames (int): Number of frames to sum over.
        @param threshold (int): Threshold for heatmap pixel values.
        @param min_bbox (int, int): Minimum (width, height) of a detection
            bounding box in pixels. Boxes smaller than this will not be drawn.
            Defaults to 2% of image size.
        @param show_video (bool): Display the video.
        @param draw_heatmap (bool): Display the heatmap in an inset in the
            upper left corner of the video.
        @param draw_heatmap_size (float): Size of the heatmap inset as a
            fraction between (0, 1) of the image size.
        @param write (bool): Write the resulting video, with detection
            bounding boxes and/or heatmap, to a video file.
        @param write_fps (num): Frames per second for the output video.
        """

        cap = video_capture
        if not cap.isOpened():
            raise RuntimeError("Error opening VideoCapture.")

        fps_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        (grabbed, frame) = cap.read()
        (h, w) = frame.shape[:2]

        # Store coordinates of all windows to be checked at every frame.
        self.windows = slidingWindow((w, h), init_size=self.init_size,
                                     x_overlap=self.x_overlap, y_step=self.y_step,
                                     x_range=self.x_range, y_range=self.y_range, scale=self.scale)

        if min_bbox is None:
            min_bbox = (int(0.02 * w), int(0.02 * h))

        # Heatmap inset size.
        inset_size = (int(draw_heatmap_size * w), int(draw_heatmap_size * h))

        if write:
            vidFilename = datetime.now().strftime("%Y%m%d%H%M") + ".avi"
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(vidFilename, fourcc, write_fps, (w, h))

        # Compute the heatmap for each frame and store in current_heatmap.
        # Store the last num_frames heatmaps in deque last_N_frames. At each
        # frame, sum in the deque to compute summed_heatmap. After
        # thresholding, label blobs in summed_heatmap with
        # scipy.ndimage.measurements.label and store in heatmap_labels.
        current_heatmap = np.zeros((frame.shape[:2]), dtype=np.uint8)
        summed_heatmap = np.zeros_like(current_heatmap, dtype=np.uint8)
        last_N_frames = deque(maxlen=num_frames)
        heatmap_labels = np.zeros_like(current_heatmap, dtype=np.int)

        # Weights for the frames in last_N_frames for producing summed_heatmap.
        # Recent frames are weighted more heavily than older frames.
        weights = np.linspace(1 / (num_frames + 1), 1, num_frames)
        bar = tqdm.tqdm(total=fps_total, desc="Detect process")
        while True:
            (grabbed, frame) = cap.read()
            bar.update()
            if not grabbed:
                bar.close()
                break

            current_heatmap[:] = 0
            summed_heatmap[:] = 0
            for (x_upper, y_upper, x_lower, y_lower) in self.classify(frame):
                current_heatmap[y_upper:y_lower, x_upper:x_lower] += 10

            last_N_frames.append(current_heatmap)
            for i, heatmap in enumerate(last_N_frames):
                cv2.add(summed_heatmap, (weights[i] * heatmap).astype(np.uint8),
                        dst=summed_heatmap)

            # Apply blur and/or dilate to the heatmap.
            # cv2.GaussianBlur(summed_heatmap, (5,5), 0, dst=summed_heatmap)
            cv2.dilate(summed_heatmap, np.ones((7, 7), dtype=np.uint8),
                       dst=summed_heatmap)

            if draw_heatmap:
                inset = cv2.resize(summed_heatmap, inset_size,
                                   interpolation=cv2.INTER_AREA)
                inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
                frame[:inset_size[1], :inset_size[0], :] = inset

            # Ignore heatmap pixels below threshold.
            summed_heatmap[summed_heatmap <= threshold] = 0

            # Label remaining blobs with scipy.ndimage.measurements.label.
            num_objects = label(summed_heatmap, output=heatmap_labels)

            # Determine the largest bounding box around each object.
            for obj in range(1, num_objects + 1):
                (Y_coords, X_coords) = np.nonzero(heatmap_labels == obj)
                x_upper, y_upper = min(X_coords), min(Y_coords)
                x_lower, y_lower = max(X_coords), max(Y_coords)

                # Only draw box if object is larger than min bbox size.
                if (x_lower - x_upper > min_bbox[0]
                        and y_lower - y_upper > min_bbox[1]):
                    cv2.rectangle(frame, (x_upper, y_upper), (x_lower, y_lower),
                                  (0, 255, 0), 6)

            if write:
                writer.write(frame)

            if show_video:
                cv2.imshow("Detection", frame)
                cv2.waitKey(1)

        cap.release()

        if write:
            writer.release()
