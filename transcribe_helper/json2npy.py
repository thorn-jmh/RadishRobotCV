from fileinput import filename
import json
import numpy as np

def toNpy(kpArr):
    npyData = np.array(kpArr).reshape(17,3)
    return npyData


def transcribeJSON2NPY(keyPointsData_JSON:json):
    frameNum = len(keyPointsData_JSON)
    outputArr = []
    frame = 0
    for frameName, data in keyPointsData_JSON.items():
        # frame = frameName.split(".jpg")[0]
        frame += 1
        version = data["version"]

        # Only work for the first person.
        # That means we can only work while
        # there is only one person.

        keypoints = data["people"][0]["pose_keypoints_2d"]
        kpNpy = toNpy(keypoints)
        outputArr.append(kpNpy)
    outputNpy = np.array(outputArr).reshape(frameNum,17,3)

    return outputNpy
    # print(outputNpy)

if __name__ == "__main__":
    keyPointsData_JSON = json.load(open("data.json","r"))
    outputNpy = transcribeJSON2NPY(keyPointsData_JSON)
    fileName = "output.npy"
    np.save(fileName, outputNpy)

