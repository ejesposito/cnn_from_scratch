import os
import cv2
import time
import numpy
from PIL import Image


def image_helper(detector, image, out_name, window_size=(100, 64)):
    start = time.time()
    detected = detector.detect(image, window_size)
    end = time.time()
    print('Detection time: {}'.format((end-start)/60))
    # draw rectangles
    image = numpy.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    nd, score, d1, d2, d3, d4, x1, x2, y1, y2, scale, i = detected
    x1 = x1 * (scale ** i)
    x2 = x2 * (scale ** i)
    y1 = y1 * (scale ** i)
    y2 = y2 * (scale ** i)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
    cv2.putText(image, '{}'.format(int(str(int(d1)) + str(int(d2)) + str(int(d3)) + str(int(d4)))),
                (int(x1), int(y2) + int(50 * (scale ** i))), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 255), 4)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    cv2.imwrite("output/{}.png".format(out_name), image)


def video_helper(video_name, fps, predictor, detector):

    video = os.path.join('test_assets', video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape
    print(image.shape)

    out_path = "output/{}".format(video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    frame_num = 1

    while image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print (image.shape)
        image = Image.fromarray(image)
        print (image.size)
        print("Processing fame {}".format(frame_num))
        start = time.time()
        detected = detector.detect(image, window_size=(100, 64), yield_first=True)
        end = time.time()
        print('Time: {}'.format((end-start)/60))
        # draw rectangles
        image = numpy.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        nd, score, d1, d2, d3, d4, x1, x2, y1, y2, scale, i = detected
        x1 = x1 * (scale ** i)
        x2 = x2 * (scale ** i)
        y1 = y1 * (scale ** i)
        y2 = y2 * (scale ** i)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, '{}'.format(int(str(int(d1)) + str(int(d2)) + str(int(d3)) + str(int(d4)))),
                    (int(x1), int(y2) + int(50 * (scale ** i))), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 2)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        video_out.write(image)
        image = image_gen.__next__()
        frame_num += 1
    video_out.release()


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)
