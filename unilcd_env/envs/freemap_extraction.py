import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
from skimage import measure


# DATA_DIR = Path("/home/tzm/Projects/2_Lab/data/carla/vi_dataset_4/Town10HD_0/")
# SAVE_DIR = DATA_DIR/"freemap/"
MIN_AREA = 100
# HEIGHT_THRESH = 1.5
WIDTH = 1920
HEIGHT = 1080
MAX_DISTANCE = 15  # meters
EPS = 10e-4
FOV = 90
FOCAL = WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))
PITCH = 0  # upward-positive
# CAM_HEIGHT = 1.5

# semantic tags


class Tags:
    Hash = {'UNLABELED': 0, 'BUILDING': 1, 'FENCE': 2, 'OTHER': 3, 'PEDESTRIAN': 4, 'POLE': 5, 'ROAD_LINE': 6, 'ROAD': 7, 'SIDEWALK': 8, 'VEGETATION': 9, 'VEHICLE': 10, 'WALL': 11, 'TRAFFIC_SIGN': 12, 'SKY': 13,
            'GROUND': 14, 'BRIDGE': 15, 'RAIL_TRACK': 16, 'GUARD_RAIL': 17, 'TRAFFIC_LIGHT': 18, 'STATIC': 19, 'DYNAMIC': 20, 'WATER': 21, 'TERRAIN': 22, 'VISUALLYIMPAIRED': 23, 'WHEELCHAIR': 24, 'WHEELPD': 25, 'CANE': 26}

    Hash2 = {'BUILDING': 1, 'PEDESTRIAN': 4, 'POLE': 5,'ROAD_LINE': 6, 'ROAD': 7, 'SIDEWALK': 8, 'VEGETATION': 9, 'VEHICLE': 10, 'WALL': 11,'GROUND': 14,'STATIC': 19}
    Colors = [(0, 0, 0), (70, 70, 70), (100, 40, 40), (55, 90, 80), (220, 20, 60), (153, 153, 153), (157, 234, 50), (128, 64, 128), (244, 35, 232), (107, 142, 35), (0, 0, 142), (102, 102, 156), (220, 220, 0), (70, 130, 180),
              (81, 0, 81), (150, 100, 100), (230, 150, 140), (180, 165, 180), (250, 170, 30), (110, 190, 160), (170, 120, 50), (45, 60, 150), (145, 170, 100), (145, 50, 164), (140, 150, 160), (20, 120, 225), (240, 140, 160)]

    FREE_SET = ['SIDEWALK', 'GROUND','ROAD']

    NONFREE_SET = ['BUILDING', 'PEDESTRIAN', 'POLE', 'VEGETATION', 'VEHICLE', 'WALL','STATIC']
    DESCRIPTION_SET = ['BUILDING', 'PEDESTRIAN', 'POLE', 'VEGETATION', 'VEHICLE', 'WALL','STATIC', 'ROAD']



class Scene():
    def __init__(self, scene_depthmap, bitmask):
        self.scene_depthmap = scene_depthmap
        self.bitmask = bitmask
        return

    def get_masked_depth(self):
        if self.bitmask is not None:
            masked_depthmap = self.bitmask * self.scene_depthmap
            return masked_depthmap

    def get_image_coordinate(self):
        masked_depthmap = self.get_masked_depth()
        # [u (image x), v (image y), depth]: with camera as origin
        roi_indeces = np.where(masked_depthmap >= EPS)
        coords = np.zeros((len(roi_indeces[0]), 3))
        if coords.shape[0] > 0:
            coords[:,0] = (roi_indeces[1] - WIDTH/2) * masked_depthmap[roi_indeces[0],roi_indeces[1]]
            coords[:,1] = (roi_indeces[0] - HEIGHT/2) * masked_depthmap[roi_indeces[0],roi_indeces[1]]
            coords[:,2] = masked_depthmap[roi_indeces[0],roi_indeces[1]]
        # for idx in range(len(roi_indeces[0])):
        #     u = roi_indeces[1][idx]
        #     v = roi_indeces[0][idx]
        #     coords[idx][0] = (u - WIDTH/2) * masked_depthmap[v][u]
        #     coords[idx][1] = (v - HEIGHT/2) * masked_depthmap[v][u]
        #     coords[idx][2] = masked_depthmap[v][u]
        return coords

    def get_local_coordinates(self, focal, pitch, height):
        ROT_M = get_rotation_mat_y(pitch)
        M = np.eye(3)
        M[0, 0] = M[1, 1] = focal
        M[0, 2] = 0.0
        M[1, 2] = 0.0
        M_inv = np.linalg.inv(M)
        image_coords = self.get_image_coordinate()
        scene_coords = image_coords @ M_inv.T
        # flip coordinates: x, y, are in opposite direction in image coordinate
        local_coords = np.concatenate(
            (scene_coords[:, 2][:, None], -scene_coords[:, 0][:, None], -scene_coords[:, 1][:, None]), axis=1)
        local_coords = local_coords @ ROT_M.T
        local_coords[:, 2] = local_coords[:, 2] + height
        return local_coords


def load_semseg(file_dir):
    image = cv2.imread(file_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth(file_dir):
    image = np.load(file_dir)
    return image


def show_semseg(image):
    plt.imshow(image)
    plt.show()
    return


def show_depth(image):
    image = np.repeat(image[:, :, None], 3, axis=2)
    image = image.astype(int)
    plt.imshow(image, vmin=0, vmax=255)
    plt.show()
    return


def get_rotation_mat_z(rot_deg):
    # clockwise
    rad = rot_deg * np.pi/180
    mat = np.zeros((3, 3))
    mat[0, 0] = np.cos(rad)
    mat[0, 1] = np.sin(rad)

    mat[1, 0] = -np.sin(rad)
    mat[1, 1] = np.cos(rad)

    mat[2, 2] = 1
    return mat


def get_rotation_mat_y(rot_deg):
    # clockwise (y-axis pointing outward)
    rad = rot_deg * np.pi/180
    mat = np.eye(3)
    mat[0, 0] = np.cos(rad)
    mat[0, 2] = -np.sin(rad)
    mat[2, 0] = np.sin(rad)
    mat[2, 2] = np.cos(rad)
    return mat


def process_scene(semseg, depth, tags, height_thresh, cam_height, rot=0, ax=None):
    depth = depth * (depth < MAX_DISTANCE)
    rot_mat = get_rotation_mat_z(rot)
    return_data = np.empty((0, 3), dtype=np.float32)
    return_tags = np.empty((0), dtype=np.int8)
    for tag in tags:
        bitmask = np.where(semseg == Tags.Colors[tag], 1, 0)
        scene = Scene(depth, bitmask[:, :, 2])
        coord = scene.get_local_coordinates(FOCAL, PITCH, cam_height)
        filtered_coord = coord[coord[:, 2] <= height_thresh]
        rotated_coord = filtered_coord @ rot_mat.T
        # data = np.concatenate((rotated_coord, np.ones((rotated_coord.shape[0], 1))*tag), axis=1)
        return_data = np.concatenate((return_data, rotated_coord), axis=0)
        return_tags = np.concatenate((return_tags, np.repeat(int(tag), rotated_coord.shape[0])))
        if ax is not None:
            ax.plot(rotated_coord[:, 0], rotated_coord[:, 1], '.', color=tuple(np.array(Tags.Colors[tag])/255))
    # print("here")
    # plt.show()
    return return_data, return_tags


def main(args):

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    tag_set = [Tags.Hash[tag] for tag in Tags.Hash][0:]
    # print(tag_set)
    N = len(os.listdir(data_dir/"annotations"))
    print("Total N: ", N)
    print("Data dir: ", data_dir)
    print("Save dir: ", save_dir)
    # for one scene
    for idx in range(N):
        file_name = '%012d' % idx
        print('Procesing: ', file_name)

        semseg = load_semseg(str(data_dir/"front_semantics"/(file_name+".png")))
        depth = load_depth(str(data_dir/"front_depth"/(file_name+"_kid.npy")))


        # fig, ax = plt.subplots()
        ax = None
        # ax.plot(0, 0, 'rx')
        # front_data = process_scene(semseg, depth, tag_set[0:], 0, ax)
        # left_data = process_scene(left_semseg, left_depth, tag_set[0:], -90, ax)
        # right_data = process_scene(right_semseg, right_depth, tag_set[0:], 90, ax)
        # back_data = process_scene(back_semseg, back_depth, tag_set[0:], 180, ax)

        front_data = process_scene(semseg, depth, tag_set[:], float(args.height_thresh), float(args.cam_height), -90, ax)

        # ax.set_ylim(-150,150)
        # ax.set_xlim(-150,150)
        # plt.show()

        # scene_data = {'town':anno['image_info']['town'],'weather':anno['image_info']['weather'],
        #             'camera_pose_info': anno['camera_pose_info'],
        #             'points': front_data[:, 0:3].astype(np.single), 'tags': front_data[:, 3].astype(np.uint8)}

        # if not os.path.exists(save_dir):
        #     save_dir.mkdir()
        # np.save(str(save_dir/(file_name+'.npy')), scene_data)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, required=True, help="Directory of raw data")
    parser.add_argument("--save_dir", default=None, required=True, help="Directory of output data")
    parser.add_argument("--height_thresh", default=1.5, required=True, help="Object height cutoff")
    parser.add_argument("--cam_height", default=None, required=True, help="Height of cameras")
    args = parser.parse_args()
    main(args)