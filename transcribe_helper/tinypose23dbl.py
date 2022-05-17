import json
import string
import os
from typing import List

from cv2 import _OutputArray_DEPTH_MASK_16F

class Frame:
    def __init__( self, fram_name:string, data:json ) -> None:
        self.name = fram_name
        self.data = []
        for i in range(17):
            self.data += data[i][:]

def makeFromatFile(frames: List[Frame]):
    opd = {}
    for i in range(len(frames)):
        frame = frames[i]
        opd[ frame.name ] = {
            "version":0.1,
            "people":[{
                "pose_keypoints_2d": frame.data
            }]
        }
    return opd
    # outputFile = open(fileName, "w")
    # outputFile.write( json.dumps(opd) )

def dataReArrange(oriData:json):
    data = oriData[2][0][0]
    outputData = data + [[]]
    outputData[0] = data[0]
    for i in range(3):
        outputData[1][i] = (data[5][i] + data[6][i]) * 0.5
    outputData[2] = data[6]
    outputData[3] = data[8]
    outputData[4] = data[10]
    outputData[5] = data[5]
    outputData[6] = data[7]
    outputData[7] = data[9]
    outputData[8] = data[12]
    outputData[9] = data[14]
    outputData[10] = data[16]
    outputData[11] = data[11]
    outputData[12] = data[13]
    outputData[13] = data[15]
    outputData[14] = data[2]
    outputData[15] = data[1]
    outputData[16] = data[4]
    outputData[17] = data[3]
    return outputData

def transcribePP23D(tJson:json):
    frames = []
    frameCnt = -1
    for data in tJson:
        frameCnt += 1
        frame = Frame(str(frameCnt), dataReArrange(data))
        frames.append(frame)
    return makeFromatFile(frames)

def departJsonData( datas:json, filepath:string):
    cnt = -1
    for data in datas:
        cnt+=1
        _out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath+'/{0}.json'.format(str(cnt)))
        with open(_out_file, 'w') as outfile:
            json.dump(data, outfile) 

if __name__ == "__main__":
    tJson = json.load(open("t.json","r"))
    output = transcribePP23D(tJson=tJson)
    print(json.dumps(output, indent=4))
