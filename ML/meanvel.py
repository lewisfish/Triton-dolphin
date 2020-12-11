from concurrent import futures
from itertools import repeat
import pathlib
from pathlib import Path
import pickle
import time
from typing import List, Tuple, Union

import cv2 as cv2
import hdbscan
import numpy as np
import pims
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


from gps import getAltitude
from ocr import getMagnification


def _getFullFileName(files: List[pathlib.PosixPath], target: str) -> pathlib.PosixPath:
    '''Match a file to list of full filenames

    Parameters
    ----------
    files : List[pathlib.PosixPath]
        List of file name to match to

    target : str
        File to be matched with full path

    Returns
    -------

    file : pathlib.PosixPath
        Full filename

    '''

    for file in files:
        if target in str(file):
            return file


def getFrames(file: str, position: int, offset: int) -> Tuple[List[pims.frame.Frame], List[int], float]:
    """Get 3 frames for optical flow analysis. Frames are serperated by +/- offset.
       Central frame is at position.

    Parameters
    ----------
    file : str
        Video file to get frames from

    position : int
        Position of central frame

    offset : int
        offset to get other frames for optical flow analysis

    Returns
    -------
    Tuple[List[pims.frame.Frame], List[int], float]
        Frames at position, +/- offset, list of frame positions, fps of video

    """

    assert position > offset
    video = pims.PyAVVideoReader(file)

    frame0 = video[position - offset]
    frame1 = video[position]
    frame2 = video[position + offset]

    return [frame0, frame1, frame2], [position - offset, position, position + offset], float(video._frame_rate)


def getFramesCV2(file: str, position: int, offset: int):
    """Get 3 frames for optical flow analysis using cv2. Frames are serperated by +/- offset.
       Central frame is at position.

    Parameters
    ----------
    file : str
        Video file to get frames from

    position : int
        Position of central frame

    offset : int
        offset to get other frames for optical flow analysis

    Returns
    -------
    Tuple[List[np.ndarray], List[int], float]
        Frames at position, +/- offset, list of frame positions, fps of video

    """

    assert position >= offset

    cap = cv2.VideoCapture(str(file))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, position-offset)
    _, frame = cap.read()
    frame0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    _, frame = cap.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.set(cv2.CAP_PROP_POS_FRAMES, position+offset)
    _, frame = cap.read()
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return [frame0, frame1, frame2], [position - offset, position, position + offset], fps


def cropFrames(frames: List[pims.frame.Frame], box: List[int]) -> List[pims.frame.Frame]:
    """Crop frames.

    Parameters
    ----------
    frames : List[pims.frame.Frame]
        List of frames to be cropped

    box : List[int]
        Dimensions, and location to crop: format [y0, x0, y1, x1]

    Returns
    -------
    List[pims.frame.Frame]
        List of cropped frames

    """

    croppedFrames = []
    xi = box[1]
    xii = box[3]
    yi = box[0]
    yii = box[2]

    for frame in frames:
        croppedFrames.append(frame[yi:yii, xi:xii])
    return croppedFrames


def preprocessFrame(frame: pims.frame.Frame, fg) -> Tuple[np.ndarray]:
    """Preprocess frame. Converts to grayscale and removes noise.

    Parameters
    ----------
    frame : pims.frame.Frame
        Frame to be preprocessed
    fg : TYPE
        Foreground remover?

    Returns
    -------
    Tuple[np.ndarray]
        Dilated image binary image, and grayscale image

    """

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fgmask = fg.apply(gray)
    blur = cv2.GaussianBlur(fgmask, (5, 5), 0)

    _, thesh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thesh, None, iterations=3)

    return dilated, gray


def processContours(contours: List[float], contourpoints: List[List[float]], frame: pims.frame.Frame, debug=False) -> Tuple[List[List[float]], pims.frame.Frame]:
    """Get bounding boxes for each contour.

    Parameters
    ----------
    contours : List[float]
        List of contours to find bounding boxes for.

    contourpoints : List[List[float]]
        List of bounding boxes. Does this need passed in?

    frame : pims.frame.Frame
        Frame from which the contours are from

    debug : bool, optional
        If true then draw bounding boxes

    Returns
    -------
    Tuple[List[List[float]], pims.frame.Frame]
        List of bounding boxes, and frame

    """

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + (w / 2)
        cy = y + (h / 2)
        contourpoints.append([cx, cy])
        if debug:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return contourpoints, frame


def processFile(file: str) -> List[Union[str, int, List[float], int]]:
    """Open, read and process data from file.

    Parameters
    ----------
    file : str
        File to read.

    Returns
    -------
    List[Union[str, int, List[float], int]]
        List of videoname, framenumber, bounding box, and label
    """

    info = []

    f = open(file, "r")
    lines = f.readlines()[1:]
    for line in lines:
        name, framenum, *box, label, vel, kmeans, hdbscan = line.split(",")
        framenum = int(framenum)
        label = int(label)
        box = [int(x) for x in box]

        item = [name, framenum, box, label]
        info.append(item)

    return info


def trainParallel(workers=8):
    """Wrapper function for training HDBSCAN in parallel.

    Parameters
    ----------
    workers : int, optional
        Number of workers to use in parallel, default=2
    """

    data = processFile("../data/train.csv")

    with futures.ProcessPoolExecutor(workers) as executor:
        res = list(tqdm(executor.map(train, data), total=len(data)))

    velocities = []
    for i in res:
        velocities.extend(i)

    # with open('velocities.npy', 'wb') as f:
    #     np.save(f, velocities)

    # model = hdbscan.HDBSCAN(min_cluster_size=1000, cluster_selection_epsilon=0.2, min_samples=5, leaf_size=100, prediction_data=True).fit(np.array(velocities).reshape(-1, 1))

    # import pickle
    # with open('model.pickle', 'wb') as f:
    #     pickle.dump(model, f)


def train(info: Tuple[str, List[float], int], root="/data/lm959/data/", crop=False):
    """Training function for HDBSCAN. Actually does the optical flow and
       returns the data needed for training.

    Parameters
    ----------
    info : Tuple[str, List[float], int]
        Tuple of video filename, framenumber, bounding box of object, and label of object.

    root : str, optional
        Root of file system location where videos are stored.

    crop : bool, optional
        If true then crop frames to bounding box of object.

    Returns
    -------
    velocitymeterPerSecond : np.ndarray
        List of velocities in m/s
    """

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    fgbg = cv2.createBackgroundSubtractorMOG2()
    root = Path(root)
    videoFiles = list(root.glob("**/*.mp4"))

    vidname, fnumber, box, label = info
    fullname = _getFullFileName(videoFiles, vidname)

    frames, framenums, fps = getFramesCV2(fullname, fnumber, offset=15)
    contourpoints = []

    fpsPerFrame = 1. / fps
    alt = getAltitude(fullname, framenums[1], gpsdataPath="../data/gps/")
    magn = getMagnification(frames[1])

    dolphLength = 1714 * (magn / alt) + 16.5
    dolphPixelPerSecond = dolphLength / 2.
    if crop:
        frames = cropFrames(frames, box)

    frame = frames[0]
    for i in range(0, 2):
        dilated, gray1 = preprocessFrame(frame, fgbg)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contourpoints, frame = processContours(contours, contourpoints, frame)
        p0 = np.array(contourpoints, np.float32)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

        try:
            p1, _, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
            diff = np.array(p1) - np.array(p0)
            velocity = diff / fpsPerFrame
            velocity = [np.sqrt(item[0]**2 + item[1]**2) for item in velocity]
            frame = frames[1].copy()
            contourpoints = []
        except:
            # velocity = np.array([0.])
            # if not crop:
            continue

    velocitymeterPerSecond = velocity / dolphPixelPerSecond
    return velocitymeterPerSecond


def calcLabels():
    """Summary
    """
    from sklearn.cluster import KMeans

    data = processFile("../data/train.csv")
    data = data
    workers = 8
    with futures.ProcessPoolExecutor(workers) as executor:
        res = list(tqdm(executor.map(train, data, repeat("/data/lm959/data/"), repeat(True)), total=len(data)))

    with open('velocities.npy', "rb") as f:
        arrays = np.load(f)
        # model = hdbscan.HDBSCAN(min_cluster_size=1000, min_samples=5, leaf_size=100, prediction_data=True).fit(arrays.reshape(-1, 1))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        TotalVelocity = scaler.fit_transform(arrays.reshape(-1, 1))
        model = KMeans(n_clusters=6, random_state=0, max_iter=300, precompute_distances=True, algorithm='full').fit(np.array(TotalVelocity).reshape(-1, 1))

    outshizz = []
    for i, item in enumerate(res):
        vels = np.mean(item)
        test_labels = model.predict(vels.reshape(-1, 1))
        tmp = [data[i][0], data[i][1], data[i][2][0], data[i][2][1], data[i][2][2], data[i][2][3], data[i][3], vels, test_labels[0]]
        outshizz.append(tmp)

    with open('train-data-kmeans.npy', 'wb') as f:
        np.save(f, np.array(outshizz))

# def infer(vels, tmper):
#     import pickle
#     with open('model.pickle', "rb") as f:
#         loaded_obj = pickle.load(f)
#         test_labels, strengths = hdbscan.approximate_predict(loaded_obj, np.array(vels).reshape(-1, 1))


def elbowPlot():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    with open('velocities.npy', "rb") as f:
        arrays = np.load(f)

        scaler = StandardScaler()
        TotalVelocity = scaler.fit_transform(arrays.reshape(-1, 1))
        inertia = []
        for i in range(1, 15):
            print(i)
            km = KMeans(n_clusters=i, random_state=0, max_iter=300, precompute_distances=True, algorithm='full')
            km.fit(np.array(TotalVelocity).reshape(-1, 1))
            inertia.append(km.inertia_)

# results = trainParallel(workers=6)
# calcLabels()
# elbowPlot()
