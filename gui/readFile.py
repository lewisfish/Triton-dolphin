import cv2


def iter_dict(videodict):
    for key in videodict:
        frames = videodict[key]["time"]
        bboxs = videodict[key]["bbox"]
        for i, j in zip(frames, bboxs):
            yield i, j


def createDict(filename):

    f = open(filename, "r")
    lines = f.readlines()
    mydict = {}
    for line in lines:
        if line[0] == "#":
            videoFile = line[1:].strip()
            mydict[videoFile] = {}
            cap = cv2.VideoCapture("../" + videoFile)  # converts to RGB by default
            fps = cap.get(cv2.CAP_PROP_FPS)  # get fps
            cap.release()
        else:
            lineSplit = line.split(",")
            hour, minute, sec = lineSplit[0].split(":")
            time = (int(hour) * 60*60) + (int(minute) * 60) + int(sec)
            frameNum = int(time * fps)

            x0 = int(lineSplit[1])
            y0 = int(lineSplit[2])
            x1 = int(lineSplit[3])
            y1 = int(lineSplit[4])
            coords = [[x0, y0], [x1, y1]]

            if "time" not in mydict[videoFile]:
                mydict[videoFile]["time"] = []
                mydict[videoFile]["bbox"] = []

            mydict[videoFile]["time"].append(frameNum)
            mydict[videoFile]["bbox"].append(coords)

    return mydict


if __name__ == '__main__':

    file = "test.dat"
    dictFrames = createDict(file)

    iterDict = iter_dict(dictFrames)
    for i in iterDict:
        print(i)
