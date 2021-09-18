import numpy as np
import cv2
from skimage import img_as_float, color
import sys
from skimage.io import imread, imshow, imsave
from sift import SIFT
from sift import my_imshow
from numpy.random import shuffle
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from math import sqrt
import os
from pathlib import Path

def draw_matches(img, mapped_to_img, points=[], mapped_points=[], scale_coeff=1, dpi=80):

        #base drawing
        figsize = scale_coeff * (img.shape[1] + mapped_to_img.shape[1]) / dpi,\
                        scale_coeff * (img.shape[0] + mapped_to_img.shape[0]) / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)

        ratio_along_y = img.shape[0] / (img.shape[0] + mapped_to_img.shape[0])
        ratio_along_x = img.shape[1] / (img.shape[1] + mapped_to_img.shape[1])
        
        ax_img = fig.add_axes([0, 1 - ratio_along_y, ratio_along_x, ratio_along_y])
        ax_img.axis([0, img.shape[1], img.shape[0], 0])
        
        ax_mapped_to_img = fig.add_axes([ratio_along_x, 0, 1 - ratio_along_x, 1 - ratio_along_y])
        ax_img,ax_mapped_to_img.axis([0, mapped_to_img.shape[1], mapped_to_img.shape[0], 0])

        ax_img.set_xticks([]), ax_img.set_yticks([])
        ax_mapped_to_img.set_xticks([]), ax_mapped_to_img.set_yticks([])

        ax_img.imshow(img, cmap="gray")
        ax_mapped_to_img.imshow(mapped_to_img, cmap="gray")

        #corresponding points drawing and connecting lines
        points = np.r_[points, points[0,:][np.newaxis, :]]
        ax_img.plot(points[:, 1], points[:, 0], color="r")

        mapped_points = np.r_[mapped_points, mapped_points[0, :][np.newaxis, :]]
        ax_mapped_to_img.plot(mapped_points[:, 1], mapped_points[:, 0], color='r')

        for i in range(points.shape[0] - 1):

                con = ConnectionPatch(xyA=points[i, ::-1], xyB=mapped_points[i, ::-1], coordsA="data", coordsB="data",\
                                        axesA=ax_img, axesB=ax_mapped_to_img, color="green")
                fig.add_artist(con)

        plt.show()

#represents line y = kx + b
class Line:

        #pair = (y, x)
        def __init__(self, pair1, pair2):

                if (pair1[1] - pair2[1]) != 0:
                
                        self.k = (pair1[0] - pair2[0]) / (pair1[1] - pair2[1])
                        self.b = pair2[0] - self.k * pair2[1]
                        self.is_conts = False

                else:
                        self.is_conts = True
                        self.const_val = pair1[1]

        def y(self, x):

                if not self.is_const:
                        return x * self.k + self.b
                else:
                        return None

        def yx(self, x, y):

                if not self.is_conts:
                        return y - self.k * x - self.b
                else:
                        return x - self.const_val 

        def all_on_one_side(self, points):

                side_values = []
                for y, x in points:
                        side_values.append(self.yx(x, y))

                side_values = np.array(side_values)

                return (side_values >= 0).all() or (side_values <= 0).all()

#pair = (y1, x1, y2, x2)
class RANSAC:

        #perspective matrix?
        #points = (y, x)

        def __init__(self, img, mapped_to_img, found_pairs,
                     sift_inst, inliers_ratio=0.41, err=10, seed=1,
                     max_iter=500):

                self.img = img
                self.mapped_to_img = mapped_to_img
                self.found_pairs = found_pairs
                self.inliers_ratio = inliers_ratio
                self.err = err
                self.max_iter = max_iter
                self.sift_inst = sift_inst
                self.seed = seed

        def is_forming_convex_polygon(self, points):

                for i in range(points.shape[0]):
                        
                        if (points[points - points[i] == 0].shape[0] > 2):
                                return False

                for i in range(points.shape[0] - 1):
                        
                        line = Line(points[i], points[i+1])

                        if not line.all_on_one_side(points):
                                return False

                return True

        #preparations to make another step of RANSAC (returning corresponding points)
        def shuffle_for_RANSAC_step(self):

                np.random.seed(self.seed)
                shuffle(self.found_pairs)

                return self.found_pairs[:4, :2], self.found_pairs[:4, 2:]

        #transforming coords using matrix
        def transform_coords(self, coords, matrix):# fix homo_w equals 0!

                homogeneous_coords = np.c_[np.roll(coords, 1, axis=1), np.ones((coords.shape[0], 1))]

                new_homo_x = np.sum(matrix[0, :] * homogeneous_coords, axis=1)
                new_homo_y = np.sum(matrix[1, :] * homogeneous_coords, axis=1)
                new_homo_w = np.sum(matrix[2, :] * homogeneous_coords, axis=1)

                new_homogeneous_coords = np.c_[new_homo_y, new_homo_x, new_homo_w]
                new_homogeneous_coords = new_homogeneous_coords / np.array(new_homogeneous_coords[:, 2])[:, np.newaxis]

                return new_homogeneous_coords[:, :2]

        def is_inside_mapped_img(self, points):
        
                if points.ndim > 1:
                
                        is_within_x_lim = (points[:, 1] >= 0).all() and (points[:, 1] < self.mapped_to_img.shape[1]).all()
                        is_within_y_lim = (points[:, 0] >= 0).all() and (points[:, 0] < self.mapped_to_img.shape[0]).all()
                
                else:

                        is_within_x_lim = (points[1] >= 0).all() and (points[1] < self.mapped_to_img.shape[1]).all()
                        is_within_y_lim = (points[0] >= 0).all() and (points[0] < self.mapped_to_img.shape[0]).all()

                return is_within_x_lim and is_within_y_lim

        def calc_inliers_ratio(self, matrix):

                inliers = 0

                for i in range(self.found_pairs.shape[0]):

                        point_from = np.r_[self.found_pairs[i, :2][::-1], 1] 
                        point_to = self.found_pairs[i, 2:][::-1]

                        RANSAC_new_point = matrix @ point_from
                        if RANSAC_new_point[2] == 0:
                                continue

                        RANSAC_new_point = RANSAC_new_point[:2] / RANSAC_new_point[2]

                        diff = point_to - RANSAC_new_point
                        if sqrt(np.sum(diff * diff)) < self.err:
                                inliers += 1
                
                return inliers
                        
        #trying to find matrix that accepts inliers_ratio
        def find_matrix(self):
                
                curr_ratio = 0
                inliers, outliers = 0, self.found_pairs.shape[0]
                max_ratio = curr_ratio
                self.limit = 0

                while (curr_ratio < self.inliers_ratio) and (self.limit < self.max_iter):

                        mapping_points, mapped_points = self.shuffle_for_RANSAC_step()
                        
                        matrix = cv2.getPerspectiveTransform(np.array(np.roll(mapping_points, 1, axis=1), dtype=np.float32),
                                                                np.array(np.roll(mapped_points, 1, axis=1), dtype=np.float32))
                        
                        

                        inliers = self.calc_inliers_ratio(matrix)
                        curr_ratio = inliers / outliers

                        if self.limit == 0:
                                max_ratio = curr_ratio
                                final_matrix = matrix
                                print(curr_ratio, inliers, outliers)

                        if curr_ratio > max_ratio:
                                final_matrix = matrix
                                max_ratio = curr_ratio
                                print(curr_ratio, inliers, outliers)

                        self.limit += 1
                        # self.sift_inst.draw_matches(img, changed_img, mapping_points, mapped_points, scale_coeff=0.1)

                # self.sift_inst.draw_matches(img, changed_img, mapping_points, mapped_points)
                return final_matrix

        #corresponding features with matrix to another features
        def get_matches(self, matrix):
                
                return np.c_[self.found_pairs[:, :2], self.transform_coords(self.found_pairs[:, :2], matrix)]

        #returning array(x1, y1, x2, y2)
        def generate_matches(self):

                perpective_matrix = self.find_matrix()
                if len(self.img.shape) == 3:
                        new_img = cv2.warpPerspective(self.img, perpective_matrix, self.mapped_to_img.shape[:-1][::-1])
                else:
                        new_img = cv2.warpPerspective(self.img, perpective_matrix, self.mapped_to_img.shape[::-1])

                return self.get_matches(perpective_matrix), new_img

#__my own version__
def rotate_to_original(dir_name):

        print('Working in ' + dir_name)
        img_names = sorted(Path(dir_name).glob('*.jpg'), key = lambda x: int(str(x).split('/')[-1].split('.')[0]))
        img_names = list(map(lambda x : str(x), img_names))
        imgs = list(map(lambda x : cv2.imread(x), img_names))
        # imgs = list(map(lambda x : cv2.resize(x, (int(x.shape[1]/4), int(x.shape[0]/4)), interpolation=cv2.INTER_CUBIC), imgs))

        os.chdir('./rotated_dataset')
        new_dir = './' + dir_name.split('/')[-1]

        if not Path(new_dir).exists():
                os.mkdir(new_dir)
        os.chdir(new_dir)

        # img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        # changed_img = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

        empty_sift_inst = SIFT(is_static=True)

        # img_features, _ = SIFT(img).get_features()
        # changed_img_features, _ = SIFT(changed_img).get_features()

        # #_original_sift_
        img = imgs[0]
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.03, sigma=1.6*3.5, nOctaveLayers = 4)
        kp, des = sift.detectAndCompute(img, None)
        img_features = np.zeros(shape=(len(des), 4), dtype=np.object)
        for i in range(img_features.shape[0]):
                        img_features[i, :] = kp[i].pt[1], kp[i].pt[0], kp[i].size, des[i].reshape(1, -1)

        print('Saved: ', img_names[0].split('/')[-1])
        cv2.imwrite(img_names[0].split('/')[-1], img)

        for img_num in range(1, len(img_names)):
                changed_img = imgs[img_num]
                kp_changed, des_changed = sift.detectAndCompute(changed_img, None)
                changed_img_features = np.zeros(shape=(len(des_changed), 4), dtype=np.object)

                for i in range(changed_img_features.shape[0]):
                        changed_img_features[i, :] = kp_changed[i].pt[1], kp_changed[i].pt[0], kp_changed[i].size, des_changed[i].reshape(1, -1)

                print(img_features.shape, changed_img_features.shape)
                # #
                found_pairs = empty_sift_inst.find_knn(changed_img_features, img_features, k=2, ratio=0.6)
                print(found_pairs.shape)
                # empty_sift_inst.draw_matches(img, changed_img, found_pairs[:, :2], found_pairs[:, 2:], scale_coeff=0.1)

                if len(found_pairs) >= 4:
                        found_pairs = np.concatenate((changed_img_features[found_pairs[:, 0], :2], img_features[found_pairs[:, 1], :2]), axis=1)
                        matched_points, RANSAC_img = RANSAC(changed_img, img, found_pairs, empty_sift_inst, inliers_ratio=1.).generate_matches()
                else:
                        RANSAC_img = np.zeros(shape=img.shape)

                # empty_sift_inst.draw_matches(img, changed_img, matched_points[:, :2], matched_points[:, 2:])
                print('Saved rotated img: ', img_names[img_num].split('/')[-1])
                cv2.imwrite(img_names[img_num].split('/')[-1], RANSAC_img)
        
        os.chdir('../..')

catalog = '/Users/macbookpro/Downloads/Ä≠®ßÆ‚‡ÆØ®Ô 08.10.2020'
sorted_folders = sorted(Path(catalog).glob('*A*'), key= lambda x: int(str(x).split('/')[-1].split('.')[0]))

for dir_name in filter(lambda x: int(str(x).split('/')[-1].split('.')[0]) > 42, sorted_folders):

        rotate_to_original(str(dir_name))

# rotate_to_original('/Users/macbookpro/Downloads/Ä≠®ßÆ‚‡ÆØ®Ô 08.10.2020/41. Apy-Sp-Ccp 18A-5')