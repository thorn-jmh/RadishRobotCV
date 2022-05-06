from bvh_skeleton import coco_skeleton
import numpy as np

if __name__ == "__main__":
    # remove them if you don't want to use.
    customPose = np.load("./input.npy")
    output = './output.bvh'

def preprocessingForNeck(customPose):
    # generate neck node
    neck = np.add(customPose[:,1],customPose[:,2])/2
    neck = np.expand_dims(neck, 1)
    customPose = np.append(customPose, neck, axis = 1)
    return customPose

def makeBvhWithNpyFile(customPose, output = "./output.bvh"):
    customPose = preprocessingForNeck(customPose)
    cc_skel = coco_skeleton.COCOSkeleton()
    cc_skel.poses2bvh(customPose, None, output)