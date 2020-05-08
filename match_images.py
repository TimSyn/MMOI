import numpy as np
import cv2
from skimage import img_as_float, color
import sys
from skimage.io import imread, imshow, imsave
from sift import SIFT
from numpy.random import shuffle
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from math import sqrt

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

class RANSAC:

        #perspective matrix?
        #points = (y, x)

        def __init__(self, img, mapped_to_img, features, mapped_to_features, inliers_ratio=0.7, err=0.5):

                self.img = img
                self.mapped_to_img = mapped_to_img
                self.features1 = features
                self.features2 = mapped_to_features
                self.inliers_ratio = inliers_ratio
                self.feature_err = err

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
        def shuffle_for_RANSAC_step(self): #try to use features

                shuffle(self.features1)

                # checkin determinant or check is quadangle or not
                while True:

                        while not self.is_forming_convex_polygon(self.features1[:4, :2]):
                                shuffle(self.features1)

                        debug_list = []
                        transform_to_points = []
                        for point in self.features1[:4, :]:

                                diff = np.concatenate(self.features2[:, 3]) - point[3]
                                closest_point = self.features2[np.argmin(np.sum(diff * diff, axis=1)), :]

                                transform_to_points.append(closest_point)
                                debug_list.append(closest_point)

                        debug_list = np.array(debug_list)
                        transform_to_points = np.array(transform_to_points)[:, :2]

                        if self.is_forming_convex_polygon(transform_to_points):# and\
                                # self.is_inside_mapped_img(transform_to_points):

                                #debug
                                feature_diff = np.concatenate(debug_list[:, 3]) - np.concatenate(self.features1[:4, 3])
                                print(np.c_[self.features1[:4, :2], transform_to_points],
                                        np.sqrt(np.sum(feature_diff * feature_diff, axis=1)), "corresponding points")

                                draw_matches(self.img, self.mapped_to_img, self.features1[:4, :2], transform_to_points)
                                #
                                return self.features1[:4, :2], transform_to_points

                        shuffle(self.features1)

        #transforming coords using matrix
        def transform_coords(self, coords, matrix): # fix homo_w equals 0!

                homogeneous_coords = np.c_[coords, np.ones((coords.shape[0], 1))]
                # new_homogeneous_coords = np.zeros(shape=homogeneous_coords.shape)

                # for i in range(coords.shape[0]):

                #         new_homogeneous_coords[i, :] = matrix @ homogeneous_coords[i, :]

                new_homo_y = np.sum(matrix[0, :] * homogeneous_coords, axis=1)
                new_homo_x = np.sum(matrix[1, :] * homogeneous_coords, axis=1)
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

        def calc_inliers_ratio(self, transformed_coords):

                founded_points_counter, inliers_points_counter = 0, 0

                for transformed_coord in transformed_coords:

                        if self.is_inside_mapped_img(transformed_coord):
                                
                                diff_arr = self.features2[:, :2] - transformed_coord[:2]
                                errors = np.sqrt(np.array(np.sum(diff_arr * diff_arr, axis=1), dtype=np.float32))
                                closest_point_index = np.argmin(errors)

                                #if errors[closest_point_index] <= sqrt(2 * self.error**2):
                                feature_diff = transformed_coord[3] - self.features2[closest_point_index, 3]
                                if sqrt(np.sum((feature_diff * feature_diff))) < self.feature_err:
                                        inliers_points_counter += 1

                                founded_points_counter += 1

                return inliers_points_counter , founded_points_counter
                                
        #trying to find matrix that accepts inliers_ratio
        def find_matrix(self):
                
                curr_ratio = 0
                inliers, outliers = 0, 0
                max_ratio = curr_ratio
                while curr_ratio < self.inliers_ratio:

                        # if curr_ratio > max_ratio:
                        print(curr_ratio, inliers, outliers)
                                # max_ratio = curr_ratio

                        mapping_points, mapped_points = self.shuffle_for_RANSAC_step()
                        
                        matrix = cv2.getPerspectiveTransform(np.array(mapping_points, dtype=np.float32),
                                                                np.array(mapped_points, dtype=np.float32))

                        #transformin coordinates from features1 to evaluate error
                        new_coords_and_features = np.c_[self.transform_coords(self.features1[:, :2], matrix), self.features1[:, 2:]] 
                        inliers, outliers = self.calc_inliers_ratio(new_coords_and_features)
                        curr_ratio = inliers / outliers
                
                print(curr_ratio)
                return matrix

        #corresponding features with matrix to another features
        def get_matches(self, matrix):
                
                return np.c_[self.features1[:, :2], self.transform_coords(self.features1[:, :2], matrix)]

        def generate_matches(self):

                perpective_matrix = self.find_matrix()

                return self.get_matches(perpective_matrix)

img, changed_img = img_as_float(color.rgb2gray(imread(sys.argv[1]))), \
                        img_as_float(color.rgb2gray(imread(sys.argv[2])))

img_features, _ = SIFT(img).get_features()
changed_img_features, _ = SIFT(changed_img).get_features()

matched_points = RANSAC(img, changed_img, img_features, changed_img_features).generate_matches()

draw_matches(img, changed_img, matched_points[:, :2], matched_points[:, 2:])

# points1 = [[1, 1], [1, img.shape[1]-1], [img.shape[0]-1, img.shape[1]-1], [img.shape[0]-1, 1]]
# points2 = [[1, 1], [1, changed_img.shape[1]-1], [changed_img.shape[0]-1, changed_img.shape[1]-1], [changed_img.shape[0]-1, 1]]

# draw_matches(img, changed_img, np.array(points1), np.array(points2))