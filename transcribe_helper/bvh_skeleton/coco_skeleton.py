from . import math3d
from . import bvh_helper

import numpy as np
class COCOSkeleton(object):
    
    def __init__(self):
        self.root = 'Hip'
        self.keypoint2index = {
            'Hip': 0,
            'L_Hip': 1,
            'L_Knee': 2,
            'L_Ankle': 3,
            'L_LEndSite': -1,
            'R_Hip': 4,
            'R_Knee': 5,
            'R_Ankle': 6,
            'R_LEndSite': -1,
            'Waist': 7,
            'Neck': 8,
            'Nose': 9,
            'Head': 10,
            'H_EndSite': -1,
            'R_Shoulder': 11,
            'R_Elbow': 12,
            'R_Wrist': 13,
            'R_HEndSite': -1,
            'L_Shoulder': 14,
            'L_Elbow': 15,
            'L_Wrist': 16,
            'L_HEndSite': -1
        }
        self.index2keypoint = {}
        for key, val in self.keypoint2index.items():
            if val != -1 :
                self.index2keypoint[val] = key
        self.keypoint_num = len(self.keypoint2index)
        self.children = {
            'Hip': [
                'L_Hip',
                'R_Hip',
                'Waist'
            ],
            'L_Hip': [ 'L_Knee' ],
            'L_Knee': [ 'L_Ankle' ],
            'L_Ankle': [ 'L_LEndSite' ],
            'L_LEndSite': [],
            'R_Hip': [ 'R_Knee' ],
            'R_Knee': [ 'R_Ankle' ],
            'R_Ankle': [ 'R_LEndSite' ],
            'R_LEndSite': [],
            'Waist': [ 'Neck' ],
            'Neck': [
                'Nose',
                'L_Shoulder',
                'R_Shoulder'
            ],
            'Nose': [ 'Head' ],
            'Head': [ 'H_EndSite' ],
            'H_EndSite': [],
            'L_Shoulder': [ 'L_Elbow' ],
            'L_Elbow': [ 'L_Wrist' ],
            'L_Wrist': [ 'L_HEndSite' ],
            'L_HEndSite': [],
            'R_Shoulder': [ 'R_Elbow' ],
            'R_Elbow': [ 'R_Wrist' ],
            'R_Wrist': [ 'R_HEndSite' ],
            'R_HEndSite': []
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent
        self.initial_directions = {
            # the T-pose
            'Hip': [0,0,0],
            'L_Hip': [0,-1,0],
            'L_Knee': [0,0,-1],
            'L_Ankle': [0,0,-1],
            'L_LEndSite': [0,0,-1],
            'R_Hip': [0,1,0],
            'R_Knee': [0,0,-1],
            'R_Ankle': [0,0,-1],
            'R_LEndSite': [0,0,-1],
            'Waist': [0,0,1],
            'Neck': [0,0,1],
            'Nose': [1,0,1],
            'Head': [-1,0,1],
            'H_EndSite': [0,0,1],
            'L_Shoulder': [0,-1,0],
            'L_Elbow': [0,-1,0],
            'L_Wrist': [0,-1,0],
            'L_HEndSite': [0,-1,0],
            'R_Shoulder': [0,1,0],
            'R_Elbow': [0,1,0],
            'R_Wrist': [0,1,0],
            'R_HEndSite': [0,1,0]
        }

    # calculate the initial offset given in HEADER section
    # according to the T-Pose defined above and the average
    # lens of each bone in the input file
    def get_initial_offset(self, poses_3d):
        # calculate the length of each bone using bfs
        bone_lens = {self.root:[0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            print(parent)
            p_idx = self.keypoint2index[parent]
            for child in self.children[parent]:
                if child.endswith('EndSite'):
                    # if the child is endsite
                    end_site_scale = {
                        'H_EndSite': 0.1,
                        'L_LEndSite': 0.4,
                        'R_LEndSite': 0.4,
                        'L_HEndSite': 0.4,
                        'R_HEndSite': 0.4
                    }
                    bone_lens[child] = end_site_scale[child] * bone_lens[parent]
                    continue

                # if the child isn't endsite
                stack.append(child)
                c_idx = self.keypoint2index[child]

                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx, :] - poses_3d[:, c_idx, :],
                    axis = 1
                )

        # bone_len is the average value of bone_lens
        bone_len = {}
        for joint in self.keypoint2index:
            # calculate the average of bone_lens
            # if its symmetrical than use both sides
            if joint.startswith('L_') or joint.startswith('R_'):
                base_name = joint.replace('L_', '').replace('R_', '')
                left_len = np.mean(bone_lens['L_' + base_name])
                right_len = np.mean(bone_lens['R_' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    # generate the HEADER section
    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = joint.endswith('EndSite')
            nodes[joint] = bvh_helper.BvhNode(
                name = joint,
                offset = initial_offset[joint],
                rotation_order = 'zxy' if not is_end_site else '',
                is_root = is_root,
                is_end_site = is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [ nodes[child] for child in children ]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root = nodes[self.root], nodes = nodes)
        return header

    # calculate the channel given in DATA section
    # according to the HEADER section and the coordination
    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]
            
            if node.is_root:
                channel.extend( pose[joint_idx] )

            index = self.keypoint2index
            order = None
            if joint == 'Hip':
                x_dir = None
                y_dir = pose[index['R_Hip']] - pose[index['L_Hip']]
                z_dir = pose[index['Waist']] - pose[joint_idx]
                order = 'zxy'
            # elif joint in ['L_Hip', 'L_Knee', 'L_Ankle']:
            #     child_idx = self.keypoint2index[node.children[0].name]
            #     x_dir = None
            #     y_dir = pose[index['Hip']] - pose[index['L_Hip']]
            #     z_dir = pose[joint_idx] - pose[child_idx]
            #     order = 'zxy'
            # elif joint in ['R_Hip', 'R_Knee', 'R_Ankle']:
            #     child_idx = self.keypoint2index[node.children[0].name]
            #     x_dir = None
            #     y_dir = pose[index['R_Hip']] - pose[index['Hip']]
            #     z_dir = pose[joint_idx] - pose[child_idx]
            #     order = 'zxy'
            # elif joint == 'L_Shoulder':
            #     x_dir = pose[index['L_Wrist']] - pose[index['L_Elbow']]
            #     y_dir = pose[joint_idx] - pose[index['L_Elbow']]
            #     z_dir = None
            #     order = 'yzx'
            # elif joint == 'L_Elbow':
            #     x_dir = pose[joint_idx] - pose[index['L_Wrist']]
            #     y_dir = pose[index['L_Shoulder']] - pose[joint_idx]
            #     z_dir = None
            #     order = 'yzx'
            # elif joint == 'R_Shoulder':
            #     x_dir = pose[index['R_Elbow']] - pose[index['R_Wrist']]
            #     y_dir = pose[index['R_Elbow']] - pose[joint_idx]
            #     z_dir = None
            #     order = 'yzx'
            # elif joint == 'R_Elbow':
            #     x_dir = pose[index['R_Shoulder']] - pose[joint_idx]
            #     y_dir = pose[joint_idx] - pose[index['R_Wrist']]
            #     z_dir = None
            #     order = 'yzx'
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()
            
            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )
            
            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel


    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)
        print(header)
        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)
        return channels, header

