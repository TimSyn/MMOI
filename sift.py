from skimage.filters import gaussian, sobel
from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import sys
import numpy as np
from skimage import img_as_float, color
from skimage.transform import resize
import random
from sklearn.preprocessing import normalize
from math import log, sqrt, cos, sin
from scipy.ndimage import convolve
import cv2
from matplotlib.patches import ConnectionPatch
from PIL import Image

#representing porabola y = ax^2 + bx + c
class Porabola:

        def __init__(self, a=0, b=0, c=0):

                self.a = a
                self.b = b
                self.c = c
        
        #pair=(x,y)
        def calc_params(self, pair1, pair2, pair3):

                A = np.array(([pair1[0]**2, pair1[0], 1], [pair2[0]**2, pair2[0], 1], [pair3[0]**2, pair3[0], 1]))
                b = np.array((pair1[1], pair2[1], pair3[1]))
                
                if np.linalg.det(A) != 0:
                        (self.a, self.b, self.c) = np.linalg.solve(A, b)
                else:
                        self.a = None
                        self.b = None
                        self.c = None

        #x coordinate of porabola vertex
        def vertex(self):
                return -self.b / (2 * self.a)

        def y(x):
                return a* x**2 + b*x + c


def my_imshow(img, dpi=80, scale_coeff=2):

        figsize = scale_coeff * img.shape[1] / dpi, scale_coeff * img.shape[0] / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(img, cmap='gray')
        plt.show()

class SIFT:
        #current problem:
                #1) fix proper img at descriptor creation / done
                #2) fit porabole /done
                #3) trilinear interpolation /done
                #4) why do i have less keypoints than cv2.SIFT/done
                #5) similar points don't have the same features

        def __init__(self, img=[], init_sigma=1.6, s=3, octave_sigma_diff_by=2, amount_of_octaves=4, threshhold=0.015, nominal_sigma=0.5,\
                        is_prior_x2_scale=True, is_static=False):

                if not is_static:
                        self.k = pow(octave_sigma_diff_by, 1.0 / s)
                        self.s = s               
                        self.img = img#_as_float(color.rgb2gray(img))
                        self.last_img = self.img.copy()
                        # self.key_point = []
                        self.min_sigma = init_sigma
                        self.is_prior_x2_scale = is_prior_x2_scale

                        if is_prior_x2_scale:
                                self.nominal_sigma = nominal_sigma * 2
                                self.octave_amount = amount_of_octaves
                        else:
                                self.nominal_sigma = nominal_sigma
                                self.octave_amount = amount_of_octaves

                        self.threshhold = threshhold
                        self.counter = 0
                        self.descr_list = []

        def show_keypoint(self, keypoints, show_scale=True, show_dots=True, all_at_once=False,\
                                show_on_DoG=True, dpi=80, scale_fig_coeff=1, DoG=[]): #second attempt

                def add_circles(centers, radius, ax, color):

                        for center in centers:
                                circle = plt.Circle((center[1], center[0]), radius, color=color, fill=False)
                                ax.add_artist(circle)

                def find_proper_img(DoG, sigma_counter):

                        scale_ind = np.argwhere(np.abs(self.sigma_comb_with_scale[:, 0] - sigma_counter) < 0.01)[0][0]
                        scale = int(self.sigma_comb_with_scale[scale_ind, 1])
                        curr_DoG = DoG[int(log(scale, 2))]
                        index = np.argwhere(np.abs(curr_DoG[:, 1] * scale - sigma_counter) < 0.01)[0][0]

                        return curr_DoG[index, 0], scale

                def create_fig_and_axe(img, scale_fig_coeff, dpi):

                        figsize = scale_fig_coeff * img.shape[1] / dpi, scale_fig_coeff * img.shape[0] / dpi
                        fig = plt.figure(figsize=figsize, dpi=dpi)
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.imshow(img, cmap='gray')
                        ax.set_xticks([]), ax.set_yticks([])
                        ax.axis([0, self.img.shape[1], self.img.shape[0], 0])

                        return fig, ax

                def create_new_fig_and_axe(img, dpi, scale_fig_coeff, sigma_counter):

                        scale = 1
                        if show_on_DoG:
                                img, scale = find_proper_img(DoG, sigma_counter)

                        img = resize(img, (scale*img.shape[0], scale*img.shape[1]))
                        fig, ax = create_fig_and_axe(img, scale_fig_coeff, dpi)

                        return fig, ax

                #transforming points to [0,1]
                keypoints[0] /= self.img.shape[0] - 1
                keypoints[1] /= self.img.shape[1] - 1

                show_on_DoG = show_on_DoG and (len(DoG) != 0)
                random.seed()
                # color = (0,0,0)
                max_sigma = np.max(keypoints[:, 2])
                
                if self.is_prior_x2_scale:
                        sigma_counter = self.min_sigma / 2
                else:
                        sigma_counter = self.min_sigma

                if all_at_once:
                        _, ax = create_fig_and_axe(self.img, scale_fig_coeff, dpi)

                # while abs(sigma_counter - max_sigma * self.k) > 0.01:
                # while max_sigma * self.k - sigma_counter > 0:
                #list[(sigma, color)]
                sigma_arr = np.array([(sigma_counter*self.k**ind, np.array((random.randint(0,100), random.randint(0,100), random.randint(0,100))) / 100)
                                for ind in range(1, self.octave_amount * (self.s + 1) + 1)])

                # print(sigma_arr.shape, sigma_arr)

                # print(keypoints)
                for keypoint in keypoints:
                        # print(keypoint[0], keypoint[1], keypoint[2], 'keypoint')

                        # color = np.array((random.randint(0,100), random.randint(0,100), random.randint(0,100))) / 100
                        # was for case if we change sigma of the keypoint
                        # bot_lim = sigma_counter - (sigma_counter - sigma_counter/self.k)/2
                        # top_lim = sigma_counter + (sigma_counter*self.k - sigma_counter)/2
                        # abs_diff = np.abs(keypoints[:, 2] - sigma_counter)
                        # centers = keypoints[(bot_lim <= abs_diff) & (abs_diff < top_lim)]
                        # centers = keypoints[np.abs(keypoints[:, 2] - sigma_counter) < 0.01]

                        # if centers.shape[0] == 0:
                        #         sigma_counter *= self.k
                        #         continue

                        ind = np.argmin(np.abs(sigma_arr[:,0] - keypoint[2]))
                        # print(ind)
                        color = sigma_arr[ind, 1]
                        radius = sigma_arr[ind, 0]
                        # print('color', color, 'radius', radius)

                        if not all_at_once:
                                _, ax = create_new_fig_and_axe(self.img, dpi, scale_fig_coeff, sigma_counter)

                        if show_scale:
                                circle = plt.Circle((keypoint[1], keypoint[0]), radius, color=color, fill=False)
                                ax.add_artist(circle)
                                #add_circles bla bla bla
                        
                        if show_dots:
                                # ax.scatter(centers[:, 1], centers[:, 0], color=color, s=1, label="Sigma is: " + "{:.1f}".format(sigma_counter))
                                ax.scatter(keypoint[1], keypoint[0], color=color, s=1)
                           
                        if not all_at_once:
                                # plt.legend()
                                plt.title("SIFT keypoints")
                                plt.show()

                        # sigma_counter *= self.k

                if all_at_once:
                        plt.legend()
                        plt.title("SIFT keypoints")
                        plt.show()

        #precomputing gradients and angles
        def grad_angles(self, img):

                extra_keypoint_enviroment_y = img[:, 1:-1]
                extra_keypoint_enviroment_x = img[1:-1, :]

                dL_y = (np.roll(extra_keypoint_enviroment_y, -1, axis=0) -\
                        np.roll(extra_keypoint_enviroment_y, 1, axis=0))[1:-1,:]

                dL_x = (np.roll(extra_keypoint_enviroment_x, -1, axis=1) -\
                        np.roll(extra_keypoint_enviroment_x, 1, axis=1))[:,1:-1]
                dL_x[dL_x == 0] = 0.0001

                angles = np.arctan2(dL_y, dL_x)
                angles[angles < 0] += 2 * np.pi
                #converting to grad
                angles = angles / (2 * np.pi) * 360

                img_copy = img.copy()
                img_copy[1:-1, 1:-1] = angles

                return img_copy
        
        #calculating the gradient magnitude
        def my_sobel(self, img):

                extra_keypoint_enviroment_y = img[:, 1:-1]
                extra_keypoint_enviroment_x = img[1:-1, :]

                dL_y = (np.roll(extra_keypoint_enviroment_y, -1, axis=0) -\
                        np.roll(extra_keypoint_enviroment_y, 1, axis=0))[1:-1,:]

                dL_x = (np.roll(extra_keypoint_enviroment_x, -1, axis=1) -\
                        np.roll(extra_keypoint_enviroment_x, 1, axis=1))[:,1:-1]

                res_gradients = img.copy()
                gradients = np.sqrt(dL_y * dL_y + dL_x * dL_x)
                res_gradients[1:-1, 1:-1] = gradients

                return res_gradients

        #list [array((array(smoothed_img), smoothing_sigma, gradient_of_smoothed_img, angles_of_smoohted_img))]
        def create_gaussian_pyramid(self):#using previous img

                #doubling the size
                sample_img = self.img.copy()
                if self.is_prior_x2_scale:
                        sample_img = cv2.resize(sample_img, (int(sample_img.shape[1]*2), int(sample_img.shape[0]*2)), interpolation=cv2.INTER_NEAREST)#resize(self.img.copy(), (int(sample_img.shape[0]*2), int(sample_img.shape[1]*2)))

                #presmoothing image to match self.min_sigma
                # sigma0 = sqrt(self.min_sigma**2 - self.nominal_sigma**2)
                # sample_img = gaussian(sample_img, sigma0)
                
                #sigmas for each sacle
                scale_sigmas = [self.min_sigma*self.k**ind for ind in range(self.s + 3)]
                print(scale_sigmas)

                #sigmas to smooth_with
                # sigma_list = []
                # for i in range(1, len(scale_sigmas)):
                #         # new_sigma = self.min_sigma*self.k**s
                #         # print(new_sigma, sqrt(new_sigma**2 - sigma_list[-1]**2))
                #         sigma_list.append(sqrt(scale_sigmas[i]**2 - scale_sigmas[i - 1]**2))

                scale_space = []
                for octave_num in range(self.octave_amount):

                        octave = []
                        for sub_sample_num in range(self.s+3):

                                if (octave_num == 0) and (sub_sample_num == 0):

                                        base_img = sample_img
                                        smoothed_img = gaussian(base_img, self.min_sigma)
                                        angles = self.grad_angles(smoothed_img)
                                        gradients = sobel(smoothed_img)
                                        octave.append((smoothed_img, self.min_sigma, gradients, angles))
                                        
                                else:

                                        if sub_sample_num == 0:

                                                # last_octave = scale_space[-1]
                                                # base_img = last_octave[self.s][0]
                                                sample_img = cv2.resize(sample_img, (int(sample_img.shape[1]/2), int(sample_img.shape[0]/2)), interpolation=cv2.INTER_NEAREST)#resize(sample_img, (int(sample_img.shape[0]/2), int(sample_img.shape[1]/2)))
                                                base_img = sample_img
                                                smoothed_img = gaussian(base_img, self.min_sigma)
                                                angles = self.grad_angles(smoothed_img)
                                                gradients = sobel(smoothed_img)
                                                octave.append((smoothed_img, self.min_sigma, gradients, angles))
                                        
                                        else:
                                                
                                                smoothed_img = gaussian(base_img, scale_sigmas[sub_sample_num])
                                                angles = self.grad_angles(smoothed_img)
                                                gradients = sobel(smoothed_img)
                                                octave.append((smoothed_img, scale_sigmas[sub_sample_num], gradients, angles)) #changed
                                
                        scale_space.append(np.array(octave))

                return scale_space

        #list [(array(DoG_itself), sigma)]
        def create_DoG_pyramid(self, gauss_pyramid):
                
                # gauss_pyramid = self.create_gaussian_pyramid()

                sigmas_in_octave = np.zeros((self.s+2, ))
                for ind in range(sigmas_in_octave.shape[0]):
                        sigmas_in_octave[ind] = self.min_sigma * self.k ** ind

                DoG = []
                for octave in gauss_pyramid:
                        
                        img_in_octave = np.array(octave[:, 0])
                        DoG_in_octave = (img_in_octave - np.roll(img_in_octave, 1, axis=0))[1:]

                        DoG.append(np.array((DoG_in_octave, sigmas_in_octave)).T)

                return DoG

        # def amend_point(self, row, col, sigma, low_lvl, mid_lvl, top_lvl, r=10)
        def amend_point(self, row, col, index, DoG_octave, r=10):

                def calc_dD_and_d2D(row, col, index):

                        row, col = int(row), int(col)
                        dD_dx = (DoG_octave[index,0][row, col+1] - DoG_octave[index,0][row, col-1]) / 2
                        dD_dy = (DoG_octave[index,0][row+1, col] - DoG_octave[index,0][row-1, col]) / 2
                        dD_ds = (DoG_octave[index,0][row, col] - DoG_octave[index,0][row, col]) / 2

                        d2D_dxdy = ((DoG_octave[index,0][row+1, col+1] - DoG_octave[index,0][row+1, col-1]) -\
                                        (DoG_octave[index,0][row-1, col+1] - DoG_octave[index,0][row-1, col-1])) / 4
                
                        d2D_dxds = ((DoG_octave[index+1,0][row, col+1] - DoG_octave[index+1,0][row, col-1]) -\
                                        (DoG_octave[index-1,0][row, col+1] - DoG_octave[index-1,0][row, col-1])) / 4

                        d2D_dyds = ((DoG_octave[index+1,0][row+1, col] - DoG_octave[index+1,0][row-1, col]) -\
                                        (DoG_octave[index-1,0][row+1, col+1] - DoG_octave[index-1,0][row-1, col])) / 4

                        d2D_d2x = DoG_octave[index,0][row, col+1] - 2*DoG_octave[index,0][row, col] + DoG_octave[index,0][row, col-1]
                        d2D_d2y = DoG_octave[index,0][row+1, col] - 2*DoG_octave[index,0][row, col] + DoG_octave[index,0][row-1, col]
                        d2D_d2s = DoG_octave[index+1,0][row, col] - 2*DoG_octave[index,0][row, col] + DoG_octave[index-1,0][row, col]

                        J = np.array([dD_dy, dD_dx, dD_ds])
                        d2J = np.array([[d2D_d2y, d2D_dxdy, d2D_dyds],
                                        [d2D_dxdy, d2D_d2x, d2D_dxds],
                                        [d2D_dyds, d2D_dxds, d2D_d2s]])
                        
                        return J, d2J

                def find_accurate_extreme(row, col, limit=3): 

                        def correct_offset(var, offset):

                                new_var = var + offset

                                if (new_var > 0).all() and (DoG_octave[index,0].shape[0] - new_var[0] > 0) and\
                                        (DoG_octave[index,0].shape[1] - new_var[1] > 0) and (DoG_octave[1,1] <= new_var[2] <=DoG_octave[-2,1]):
                                        return offset
                                else:
                                        return np.array([0,0,0])

                        var = np.array([row, col, DoG_octave[index,1]])
                        dDdvar, d2Dd2var = calc_dD_and_d2D(*np.rint(var[:2]), index)
                        var_offset = correct_offset(var, (-np.linalg.inv(d2Dd2var)).dot(dDdvar))
                        D = DoG_octave[index, 0][row, col]
                        counter = 0

                        new_index = index
                        while (np.abs(var_offset) >= 0.5).any() and (counter < limit):#dont forget to throw away bad offset

                                D = D + (1/2) * (dDdvar.T).dot(var_offset)
                                
                                if (var_offset[2] >= 0.5) and (new_index != DoG_octave.shape[0]-2):
                                        new_index += 1
                                if (var_offset[2] <= -0.5) and (new_index != 1):
                                        new_index -= 1

                                var += var_offset
                                dDdvar, d2Dd2var = calc_dD_and_d2D(*np.rint(var[:2]), new_index)
                                
                                if np.linalg.det(d2Dd2var) == 0:
                                        break
                                
                                var_offset = correct_offset(var, (-np.linalg.inv(d2Dd2var)).dot(dDdvar))
                                counter += 1


                        return var, d2Dd2var[0:2, 0:2], D
                        
                new_var, Hessian, D_at_point = find_accurate_extreme(row, col)

                # if (abs(D_at_point) < self.threshhold) or\
                if ((Hessian.trace()**2 / np.linalg.det(Hessian)) >= ((r + 1)**2 / r)):
                        return -1, -1

                return new_var#np.array((new_var[0], new_var[1], sigma))

        # def find_all_extremes(self, low_lvl, mid_lvl, top_lvl, mid_sigma, DoG_octave, env=16):  # var = (y, x, sigma)
        def find_all_extremes(self, index, DoG_octave, env=20):

                def check_extr(row, col, mid_val):

                        env_diff = np.array((DoG_octave[index + 1,0][row-1:row+2, col-1:col+2] - mid_val,\
                                                DoG_octave[index,0][row-1:row+2, col-1:col+2] - mid_val,\
                                                DoG_octave[index - 1,0][row-1:row+2, col-1:col+2] - mid_val))
                        is_min, is_max = False, False

                        #check if max
                        env_diff[1, 1, 1] = -1
                        is_max = (env_diff < 0).all()

                        #check if min
                        env_diff[1, 1, 1] = 1
                        is_min = (env_diff > 0).all()

                        return is_max or is_min

                def debug_draw(img, point, radius, dpi=80):

                        figsize = img.shape[1] / dpi, img.shape[0] / dpi
                        fig = plt.figure(figsize=figsize, dpi=dpi)
                        ax = fig.add_axes([0,0,1,1])
                        ax.set_xticks([]), ax.set_yticks([])
                        ax.axis([0, img.shape[1], img.shape[1], 0])
                        ax.imshow(img, cmap='gray')
                        # ax.scatter(point[1], point[0], color='r', s=1)
                        circle = plt.Circle((point[1], point[0]), radius, color='yellow', fill=False)
                        ax.add_artist(circle)
                        plt.show()

                extreme_points = []
                # good_points = DoG_octave[index, 0][DoG_octave[index,0]]
                for row in range(env // 2, DoG_octave[index, 0].shape[0] - 1 - env // 2):

                        for col in range(env // 2, DoG_octave[index, 0].shape[1] - 1 - env // 2):

                                if (abs(DoG_octave[index, 0][row, col]) > self.threshhold) and check_extr(row, col, DoG_octave[index, 0][row, col]):

                                        # var = (row, col, DoG_octave[index, 1])
                                        # debug_draw(DoG_octave[index, 0], (row, col), DoG_octave[index, 1])
                                        var = self.amend_point(row, col, index, DoG_octave)#, mid_sigma, low_lvl, mid_lvl, top_lvl)
                                        if (np.array(var) >= 0).all():

                                                extreme_points.append(var)

                return extreme_points

        #list of blocks (keypoints, sigma, scale_value_to_get_back_normal)
        def find_and_filter_key_points(self, DoG):
                
                keypoints = []

                if self.is_prior_x2_scale:
                        img_scale = 0.5
                else:
                        img_scale = 1

                
                for DoG_octave in DoG:

                        # for index in range(DoG_octave.shape[0]):
                        for index in range(1, DoG_octave.shape[0] - 1):
                                
                                extremes = self.find_all_extremes(index, DoG_octave)

                                # print('scale=', img_scale, 'index=', index, 'max intens=', DoG_octave[index, 0].max(), 'min instense=', DoG_octave[index, 0].min())
                                if len(extremes) != 0:
                                        print(len(extremes))
                                        extremes = np.array(extremes)
                                        keypoints.append((extremes, DoG_octave[index, 1], img_scale))
                                # img = Image.fromarray(DoG_octave[index, 0])
                                # img.save('DoG_with_scale=' + str(img_scale) + '_index=' + str(index) + '_max='+\
                                #         str(DoG_octave[index, 0].max())+'_min='+str(DoG_octave[index,0].min())+'_sigma=' + str(DoG_octave[index, 1])+'.tif', format='TIFF')

                        img_scale *= 2

                return keypoints

        #finding proper gauss pyramid element to perform descriptor calculations
        def find_proper_gauss_pyr_elem(self, sigma, Gauss_block):

                if sigma <= Gauss_block[1, 1]:
                        return Gauss_block[1]

                if sigma >= Gauss_block[self.s, 1]:
                        return Gauss_block[self.s]

                for guass_elem_counter in range(1, self.s+1):

                        if sigma > Gauss_block[guass_elem_counter+1, 1]:
                                continue

                        middle = (Gauss_block[guass_elem_counter + 1, 1] + Gauss_block[guass_elem_counter, 1]) / 2

                        if sigma <= middle:
                                return Gauss_block[guass_elem_counter]
                        
                        else:
                                return Gauss_block[guass_elem_counter + 1]

        #generating feature for one point
        def create_keypoint_features(self, keypoint, gradient, grad_angles, gauss_pyr_elem, moving_aver_param=3, ma_convolve_amount=3, amount_of_bins=36):

                #fixed gaussian kernal to use it as coeffs
                def fixed_gauss_kernel(sigma, size): 
        
                        x , y = np.mgrid[-size:size + 1, -size:size + 1]
                        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2*np.pi*sigma**2)
                        
                        return kernel / kernel.sum()

                def build_histrogram(gradient_env, angles_env, amount_of_bins):
                
                        histogram = np.zeros((amount_of_bins,))

                        #creating histogram index from angles
                        angles_env = np.trunc(angles_env / (360 / amount_of_bins))

                        #calculating histogram itself
                        for i in range(angles_env.shape[0]):
                                for j in range(angles_env.shape[1]):

                                        histogram[int(angles_env[i, j])] += gradient_env[i, j]

                        #converting to [0, 1]
                        # histogram /= histogram.max()

                        return histogram

                #prepering gradient by multiplying by Gauss kernel and angles
                def sample_grad_and_angles(gradient, grad_angles, keypoint, offset, sigma):
                                                                                        #get the closest Image with scale
                        #is enought place to take 16*16 enviroment?
                        def check_descr_cond(keypoint, descriptor_window_width=16):

                                x, y, radius = np.ceil(keypoint[1]), np.ceil(keypoint[0]), np.ceil(descriptor_window_width*sqrt(2))

                                return (gradient.shape[1] - x > radius) and (x > radius) and\
                                        (gradient.shape[0] - y > radius) and (y > radius)

                        (k_y, k_x, _) = int(keypoint[0]), int(keypoint[1]), keypoint[2]

                        if offset == 0:
                                offset = max((int(np.ceil(3 * sigma)), 3))

                        #smoothing gradients
                        gradients = gradient[k_y-offset:k_y+offset+1, k_x-offset:k_x+offset+1].copy()

                        if (gradients.size != (2*offset+1) ** 2) or not check_descr_cond(keypoint):
                                return np.array([]), np.array([])

                        gradients = gradients * fixed_gauss_kernel(sigma, offset)
                        angles = grad_angles[k_y-offset:k_y+offset+1, k_x-offset:k_x+offset+1].copy()

                        return gradients, angles

                #building subregion and descriptor itself
                def add_descriptor(keypoint_with_ori, gauss_pyr_elem, descriptor_window_width=16, desc_sigma=8, hist_width=4, hist_bins=8):

                        #normilized to unit vector
                        def convert_sub_hists_to_desc(histograms):

                                descriptor = histograms[0, 0]
                                for hist in histograms[1:, :]:
                                        descriptor = np.r_[descriptor, hist[0]]

                                descriptor /= sqrt(np.sum(descriptor * descriptor))
                                #just like Lowe says
                                # norm = sqrt(np.sum(descriptor * descriptor))
                                descriptor = np.clip(descriptor, 0, 0.2)

                                return descriptor / sqrt(np.sum(descriptor * descriptor))

                        #creating hist with shape (hist_width, hist_width) and assign it's coordinate respecting to keypoint
                        #array of rows = (sub_hist_array, sub_hist_coords_array)
                        def create_sub_regions(hist_width, hist_bins, descriptor_window_width):

                                sub_histograms_array = np.array((np.zeros((hist_bins, )), (np.array((0,0))))).reshape(1,2)

                                interval = (descriptor_window_width + 1) / (descriptor_window_width / hist_width + 1)
                                coords = np.array([[(-3*interval/2, -3*interval/2), (-interval/2, -3*interval/2), (interval/2, -3*interval/2), (3*interval/2, -3*interval/2)], 
                                                        [(-3*interval/2, -interval/2), (-interval/2, -interval/2), (interval/2, -interval/2), (3*interval/2, -interval/2)],
                                                        [(-3*interval/2, interval/2), (-interval/2, interval/2), (interval/2, interval/2), (3*interval/2, interval/2)],
                                                        [(-3*interval/2, 3*interval/2), (-interval/2, 3*interval/2), (interval/2, 3*interval/2), (3*interval/2, 3*interval/2)]])
                                #j=x , i=y
                                for i in range(hist_width):

                                        for j in range(hist_width):

                                                new_sub_hist = np.array((np.zeros((hist_bins,)), coords[i, j])).reshape(1,2)
                                                sub_histograms_array = np.concatenate((sub_histograms_array, new_sub_hist), axis=0)

                                return sub_histograms_array[1:, :]

                        #debug function
                        def debug_draw(img, points, scale_coeff=1, dpi=80):

                                img = img.copy()
                                # img = resize(self.img.copy(), (int(sample_img.shape[0]*2), int(sample_img.shape[1]*2)))
                                figsize = img.shape[1] * scale_coeff / dpi, img.shape[0] * scale_coeff / dpi
                                fig = plt.figure(figsize=figsize, dpi=dpi)
                                ax = fig.add_axes([0, 0, 1, 1])
                                ax.imshow(img, cmap='gray')
                                ax.axis([0, img.shape[1], img.shape[0], 0])
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.scatter(points[1:5, 0], points[1:5, 1], color='yellow',s=1)
                                ax.scatter(points[5:, 0], points[5:, 1], color='r', s=1)
                                ax.scatter(points[0,0], points[0, 1], color='b', s=15)
                                plt.show()

                        #debug function
                        def plot_histograms(array_of_hist, width=1400, height=800, dpi=80, bins=8):

                                figsize = width / 80, height / 80
                                # fig = plt.figure(figsize=figsize, dpi=dpi)

                                fig, plots = plt.subplots(4, 4, figsize=figsize, dpi=dpi)
                                
                                x = [bin_n * 180/bins for bin_n in range(1, 2 * bins, 2)]

                                counter = 0
                                for i in range(4):
                                        for j in range(4):

                                                plots[i, j].bar(x, array_of_hist[counter], width=7)
                                                plots[i, j].set_xticks(x)
                                                plots[i, j].tick_params('x', labelrotation=30)
                                                counter += 1

                                plt.show()

                        #debug function
                        def compare_descr(des, width=1400, height=400, dpi=80, color='r'):

                                figsize = width / 80, height / 80
                                fig = plt.figure(figsize=figsize, dpi=dpi)
                                ax = fig.add_axes([0.1, 0.1, 1, 1])

                                x = list(range(1, 129))
                                ax.bar(x, list(des), color=color)
                                ax.set_xticks(x)
                                ax.tick_params('x', labelrotation=30)


                        def assign_value(sub_hists_arr, descriptor_window_width, keypoint_with_ori, hist_width, hist_bins):

                                def gauss_weight(x, y, sigma):
                                        return np.exp(-(x**2 + y**2) / (2 * sigma**2))# / (2 * np.pi * sigma**2)

                                orientation = keypoint_with_ori[3]
                                axis_hist_distance = (descriptor_window_width + 1) / (descriptor_window_width / hist_width + 1)
                                rad_ori = orientation / 360 * 2 * np.pi
                                Rot_matr = np.array([[cos(rad_ori), -sin(rad_ori)],
                                                        [sin(rad_ori), cos(rad_ori)]])
                                orient_point =  Rot_matr @ np.array([10,0]) + keypoint_with_ori[:2][::-1]
                                # print(orientation)
                                # print(Rot_matr)
                                points_for_debug_rot =[]
                                points_for_debug = []
                                points_for_debug.append(orient_point)
                                points_for_debug_rot.append(orient_point)
                                points_for_debug_rot.append([keypoint_with_ori[1], keypoint_with_ori[0]])
                                for i in range(-descriptor_window_width//2, descriptor_window_width//2 + 1):

                                        for j in range(-descriptor_window_width//2, descriptor_window_width//2 + 1):

                                                #rotationg sample coordinates rot_coord = (x, y)
                                                rot_sample_coords = Rot_matr @ np.array([j, i])

                                                #finding closest sub_histograms to the sample to interpolate
                                                closest_hists = []
                                                for sub_hist in sub_histograms_array:
                                                        if (np.abs(sub_hist[1] - np.array([j, i])) < axis_hist_distance).all():
                                                                closest_hists.append(sub_hist)

                                                #tri-linear interpolation
                                                #interpolate values to the closest sub_hists of the sample
                                                bin_width = 360 / hist_bins
                                                y, x = int(round(rot_sample_coords[1] + keypoint_with_ori[0])),\
                                                        int(round(rot_sample_coords[0] + keypoint_with_ori[1]))

                                                points_for_debug_rot.append((x, y))
                                                points_for_debug.append((keypoint_with_ori[1] + j, keypoint_with_ori[0] + i))

                                                sample_magn = gradient[y, x] * gauss_weight(j, i, desc_sigma)
                                                sample_angle = (grad_angles[y, x] - orientation) % 360
                                                closest_hists_amount = len(closest_hists)
                                                # print(axis_hist_distance, 'axis_hist_distance')
                                                #how to assign weights to samples around the border of descriptor window?
                                                for sub_hist in closest_hists: 

                                                        coords_diff = np.abs(sub_hist[1] - [j, i])
                                                        #sample is inside
                                                        # print(closest_hists_amount, "closest_hist_amount")
                                                        if closest_hists_amount == 4:

                                                                x_weight = 1 - coords_diff[0] / axis_hist_distance
                                                                y_weight = 1 - coords_diff[1] / axis_hist_distance

                                                        #sample around border
                                                        if closest_hists_amount == 2:

                                                                #define where smaple is (along x border or along y border)
                                                                hist_coord_diff = closest_hists[0][1] - closest_hists[1][1]
                                                                if hist_coord_diff[0] == 0:
                                                                        x_weight = 1
                                                                        y_weight = 1 - coords_diff[1] / axis_hist_distance
                                                                else:
                                                                        x_weight = 1 - coords_diff[0] / axis_hist_distance
                                                                        y_weight = 1

                                                        #sample at corner
                                                        if closest_hists_amount == 1:

                                                                x_weight = 1
                                                                y_weight = 1
                                                        # print("new sample")
                                                        for bin_ori in range(hist_bins):

                                                                center_ori_in_bin = bin_ori * bin_width + bin_width / 2
                                                                if (abs(center_ori_in_bin - (sample_angle + bin_width / 2) % 360) < bin_width / 2) or\
                                                                        (abs(center_ori_in_bin - (sample_angle - bin_width / 2) % 360) < bin_width / 2):

                                                                        angle_diff = abs(center_ori_in_bin - sample_angle)
                                                                        if angle_diff < bin_width:
                                                                                angle_weight = 1 - angle_diff / bin_width
                                                                        else:
                                                                                angle_weight = 1 - (360 - angle_diff) / bin_width
                                                                        sub_hist[0][bin_ori] += sample_magn * x_weight * y_weight * angle_weight

                                                                        # if sample_magn * x_weight * y_weight * angle_weight < 0:
                                                                        #         print(x_weight, y_weight, angle_weight, sample_magn * x_weight * y_weight * angle_weight)
 
                                                # print('orientation=', keypoint_with_ori[3])
                                                # print('coords=',(j,i), 'closest_hist_amount=', closest_hists_amount, 'sample angle=',sample_angle)
                                                # plot_histograms(sub_histograms_array[:, 0])

                                # print("new")
                                # print(sub_histograms_array)
                                # keypoint_with = np.array(keypoint_with_ori[:2])
                                # p = np.array([125, 230])
                                # # if ((np.abs(keypoint_with * 0.5 - p)< 2).all() or (np.abs(keypoint_with - p) < 2).all() or\
                                # #         (np.abs(keypoint_with * 2 - p) < 2).all() or (np.abs(keypoint_with * 4 - p) < 2).all()):
                                # if (keypoint_with[1] == 115.0) and (keypoint_with[0] == 62.0):
                                #         print(keypoint_with_ori)
                                #         debug_draw(gauss_pyr_elem[0], np.array(points_for_debug))
                                #         debug_draw(gauss_pyr_elem[0], np.array(points_for_debug_rot))
                                #         plot_histograms(sub_histograms_array[:, 0])

                                # p = np.array([200, 178])
                                # if ((np.abs(keypoint_with * 0.5 - p)< 1).all() or (np.abs(keypoint_with - p) < 1).all() or\
                                #         (np.abs(keypoint_with * 2 - p) < 1).all() or (np.abs(keypoint_with * 4 - p) < 1).all()):
                                # if (keypoint_with[1] == 178.19266031568677) and (keypoint_with[0] == 199.46211832224307):
                                #         print(keypoint_with_ori)
                                #         debug_draw(gauss_pyr_elem[0], np.array(points_for_debug))
                                #         debug_draw(gauss_pyr_elem[0], np.array(points_for_debug_rot))
                                #         plot_histograms(sub_histograms_array[:, 0])
                                
                                return sub_histograms_array

                        sub_histograms_array = create_sub_regions(hist_width, hist_bins, descriptor_window_width)
                        assign_value(sub_histograms_array, descriptor_window_width, keypoint_with_ori, hist_width, hist_bins)
                        descriptor = convert_sub_hists_to_desc(sub_histograms_array).reshape(1, -1)

                        # if (keypoint_with_ori[0] == 39.403853292300525) and (keypoint_with_ori[1] == 62.090737184964894):
                        #         compare_descr(descriptor, color='r')

                        # if (keypoint_with_ori[0] == 61.86978715657008) and (keypoint_with_ori[1] == 31.477805461118066):
                        #         compare_descr(descriptor, color='y')
                        
                        return (*keypoint_with_ori[:3], descriptor)
                
                #finding accurate rotation by fitting porabola (n_left, hist[n_left]) (n_max, hist[n_max]) (n_right, hist[n_right])
                def fit_porabola(n_max, hist):
                        
                        #indeces where we take hist values
                        n_right = (n_max + 1) % hist.size
                        n_left = (n_max - 1) % hist.size

                        bin_index_equivalent = 360 / hist.size
                        n_left_or = (n_max - 1) * bin_index_equivalent + 0.5 * bin_index_equivalent
                        n_max_or = n_max * bin_index_equivalent + 0.5 * bin_index_equivalent
                        n_right_or = (n_max + 1) * bin_index_equivalent + 0.5 * bin_index_equivalent

                        porabola = Porabola()
                        porabola.calc_params((n_left_or, hist[n_left]), (n_max_or, hist[n_max]), (n_right_or, hist[n_right]))

                        #checking limits
                        clipped_interpolated_ori = porabola.vertex()
                        if (clipped_interpolated_ori < n_max_or - bin_index_equivalent / 2) or\
                                (clipped_interpolated_ori > n_max_or + bin_index_equivalent / 2):
                                clipped_interpolated_ori = n_max_or

                        return clipped_interpolated_ori % 360

                #smothing hist
                def smooth_hist(histogram, kernel):

                        tmp_histogram = np.pad(histogram, kernel.shape[0]//2)
                        tmp_histogram[0] = tmp_histogram[-3]
                        tmp_histogram[1] = tmp_histogram[-4]
                        tmp_histogram[-1] = tmp_histogram[2]
                        tmp_histogram[-2] = tmp_histogram[3]

                        new_hist = np.zeros(histogram.shape)
                        for i in range(new_hist.shape[0]):
                                new_hist[i] = tmp_histogram[i:i+kernel.shape[0]] @ kernel

                        return new_hist

                #creating main histogram to define orientation
                main_gradient, main_angles = sample_grad_and_angles(gradient, grad_angles, keypoint, 0, 1.5*keypoint[2])

                if main_gradient.size == 0:
                        return []

                main_histogram = build_histrogram(main_gradient, main_angles, amount_of_bins)

                #smooth
                # kernel = np.array([1, 4, 6, 4, 1])
                # main_histogram = smooth_hist(main_histogram, kernel=(kernel/kernel.sum()))

                #moving average
                # for _ in range(ma_convolve_amount):
                #         main_histogram = convolve(main_histogram, np.ones((moving_aver_param,)) / moving_aver_param, mode='nearest')

                # print('new_orientation')
                new_keypoints = [(keypoint[0], keypoint[1], keypoint[2], fit_porabola(orient_ind[0], main_histogram))
                                        for orient_ind in np.argwhere(main_histogram >= 0.8 * main_histogram.max())]

                # print(np.argwhere(main_histogram >= 0.8*main_histogram.max()) *10)
                # print('new point', keypoint[:2], 'orient', len(new_keypoints))
                features = [add_descriptor(new_keypoint, gauss_pyr_elem=gauss_pyr_elem) for new_keypoint in new_keypoints]

                return features
                                
        #finding prorper gaussian block within that we search proper smoothed img
        def find_gauss_block(self, scale, GaussPyr):

                if int(scale) == 0:
                        return GaussPyr[0]
                
                if self.is_prior_x2_scale:
                        index = int(log(scale, 2) + 1)
                else:
                        index = int(log(scale, 2))

                return GaussPyr[index]

        # keypoints = list[(keypoints_with_feature), sigma, scale]
        def create_features_itself(self, keypoints, GaussPyr):
                
                # for i in [1,2]:
                        # features = []
                        # outside_pyram_counter = 0       #upper limit self.octave_amount - 1
                        # inside_pyram_counter = 0        #upper limit self.s
                        # for keypoint_block in keypoints:

                        #         feature_block = []
                        #         while True:

                        #                 if (inside_pyram_counter > self.s):
                        #                         inside_pyram_counter = 0
                        #                         outside_pyram_counter += 1

                        #                 curr_gauss_block = GaussPyr[outside_pyram_counter]
                        #                 gauss_pyr_elem = curr_gauss_block[inside_pyram_counter]

                        #                 if (keypoint_block[1] == gauss_pyr_elem[1]):

                        #                         current_keypoints = keypoint_block[0]

                        #                         for keypoint in current_keypoints:
                        #                                 # if (self.is_good_env(keypoint, gauss_pyr_elem[2])):
                        #                                         feature_block += self.create_keypoint_features(keypoint, gauss_pyr_elem[2], gauss_pyr_elem[3])
                                                
                        #                         inside_pyram_counter += 1
                        #                         break

                        #                 else:
                        #                         inside_pyram_counter += 1
                                
                        #         features.append((np.array(feature_block), keypoint_block[1], keypoint_block[2]))
                                
                        # return features

                features = []
                for keypoint_block in keypoints:

                        feature_block = []
                        Gauss_Block = self.find_gauss_block(keypoint_block[2], GaussPyr)
                        for keypoint in keypoint_block[0]:
                                
                                        gauss_pyr_elem = self.find_proper_gauss_pyr_elem(keypoint[2], Gauss_Block)
                                        returned_features = self.create_keypoint_features(keypoint, gauss_pyr_elem[2], gauss_pyr_elem[3], gauss_pyr_elem)
                                        
                                        if len(returned_features) > 0:
                                                feature_block += returned_features
                        
                        if len(feature_block) > 0:
                                features.append((np.array(feature_block), keypoint_block[1], keypoint_block[2]))
                        
                return features

        #additionaly creating list[(sigma, corresponding_scale)] to show_on_DoG
        def rescaling_keypoints_back(self, keypoints):

                new_kepoints = []
                self.sigma_comb_with_scale = []
                for keypoint_block in keypoints:

                        self.sigma_comb_with_scale.append((keypoint_block[1] * keypoint_block[-1],
                                                                keypoint_block[-1]))

                        new_kepoints.append(np.c_[np.array(keypoint_block[0][:, :3] * keypoint_block[-1]), keypoint_block[0][:, 3]])

                self.sigma_comb_with_scale = np.array(self.sigma_comb_with_scale)

                return new_kepoints

        def get_features(self):

                def create_solid_arr(keypoints):

                        new_arr = keypoints[0]
                        for key_bl in keypoints[1:]:
                                new_arr = np.r_[new_arr, key_bl]

                        return new_arr

                GaussPyr = self.create_gaussian_pyramid()
                print("Pyr done")
 
                DoG = self.create_DoG_pyramid(GaussPyr)
                print("Dog done")
 
                keypoints = self.find_and_filter_key_points(DoG)
                print("Extrema found")
 
                keypoints_with_features = self.create_features_itself(keypoints, GaussPyr)
                print("Features created")
 
                keypoints_res = self.rescaling_keypoints_back(keypoints_with_features)
                print("Rescaled")

                arr = create_solid_arr(keypoints_res)
                print(arr.shape, self.counter)

                return arr, DoG

        def draw_matches(self, img, mapped_to_img, points=[], mapped_points=[], scale_coeff=1, dpi=80):
                
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
                # points = np.r_[points, points[0,:][np.newaxis, :]]
                # print(points.shape, mapped_points)
                # if (points.shape[0] != 1) and (mapped_points.shape[0] != 1):
                ax_img.scatter(points[:, 1], points[:, 0], color="r")
                ax_mapped_to_img.scatter(mapped_points[:, 1], mapped_points[:, 0], color='r')
                # else:
                        # ax_img.scatter(points[0], points[1], color="r")
                        # ax_mapped_to_img.scatter(mapped_points[0], mapped_points[1], color='r')

                for i in range(points.shape[0]):

                        con = ConnectionPatch(xyA=points[i, ::-1], xyB=mapped_points[i, ::-1], coordsA="data", coordsB="data",\
                                                axesA=ax_img, axesB=ax_mapped_to_img, color="green")
                        fig.add_artist(con)

                plt.show()

        def build_matches(self, set1, set2):

                def find_close_point_ind_by_ind(set_where_we_find, set_index_from, index, ratio=0.9):

                        point = set_index_from[index, :]
                        diff = np.concatenate(set_where_we_find[:, 3]) - point[3]
                        diff = np.sqrt(np.sqrt(np.sum(diff * diff, axis=1)))
                        closest_index = np.argmin(diff)

                        closest_distance = diff[closest_index]
                        diff[closest_index] = diff.max() + 3
                        second_close_distance = diff[np.argmin(diff)]

                        if closest_distance / second_close_distance > ratio:
                                return -1, 0
                        else:
                                return closest_index, closest_distance

                #i, j, i1 - indeces
                #i from pairs1 -> j from pairs2
                #j -> i1 from pairs1
                # i =?= i1
                found_matches = np.zeros((1, 5))
                # for i in range(set1.shape[0]):
                while True:

                        pair = np.array([])
                        for i in range(set1.shape[0]):


                                j, dist1 = find_close_point_ind_by_ind(set2, set1, i)
                                i1, dist2 = find_close_point_ind_by_ind(set1, set2, j)

                                # p1 = np.array([199.46211832224307, 178.19266031568677])
                                if (i1 == i):# and ((p - set1[i, :2] == 0).all() or (p - set2[i, :2] == 0).all()):
                                        pair = np.r_[set1[i, :2], set2[j, :2], dist1].reshape(1,5)

                                        if i == 1:
                                                set1 = set1[1:, :]
                                        if i == set1.shape[0] - 1:
                                                set1 = set1[:-1, :]
                                        else:
                                                set1 = np.concatenate((set1[:i-1, :], set1[i+1:, :]))

                                        if j == 1:
                                                set2 = set2[1:, :]
                                        if j == set2.shape[0] -1:
                                                set2 = set2[:-1, :]                                        
                                        else:
                                                set2 = np.concatenate((set2[:j-1, :], set2[j+1:, :]))

                                        found_matches = np.concatenate((found_matches, pair))
                                        break
                        
                        if pair.size == 0:
                                break

                sorted(found_matches, key=lambda pair: pair[4])
                return np.array(found_matches[1:50, :4])


sift_inst = SIFT(img_as_float(color.rgb2gray(imread(sys.argv[1]))), is_prior_x2_scale=True)
img_keypoints_and_features, DoG = sift_inst.get_features()
sift_inst.show_keypoint(img_keypoints_and_features, all_at_once=True, show_scale=True, show_dots=False, scale_fig_coeff=1, DoG=DoG, show_on_DoG=True)