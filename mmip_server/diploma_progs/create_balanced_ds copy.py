import numpy as np
from skimage.io import imread, imshow, imsave
from pathlib import Path
from skimage.filters import sobel, gaussian
from skimage.feature import canny
from skimage.color import rgb2gray
import cv2
import os
import json

def find_valid_part(filtered_img):

    def get_valid_line(line):

        length = len(line)
        ind = np.where(line)[0]
        if len(ind) > 0:
            valid_line = np.ones(shape=(length,))
            valid_line[:ind[0]] = 0
            valid_line[:-(length - ind[-1] + 1):-1] = 0
        else:
            valid_line = np.zeros(shape=(length,))
            
        return valid_line

    img_row_fill = np.ones(filtered_img.shape)
    img_col_fill = np.ones(filtered_img.shape)

    rows_num, cols_num = filtered_img.shape
    for num in range(rows_num):
        # if num < rows_num:
            img_row_fill[num] = get_valid_line(filtered_img[num])
        # if num < cols_num:
        #     img_col_fill[:, num] = get_valid_line(filtered_img[:, num])
    
    # cv2.imwrite(str(num) + '.png', img_row_fill.astype(np.uint8)*255)
        
    return img_row_fill

def connect_borders(img, pad=10):

    img = img.copy()
    indeces_x, indeces_y = np.where(img)
    x_t_s, x_b_s = indeces_x.min(), indeces_x.max()
    y_l_s, y_r_s = indeces_y.min(), indeces_y.max()
    ind_top, ind_bot = np.where(img[x_t_s])[0], np.where(img[x_b_s])[0] 
    ind_left, ind_right = np.where(img[:, y_l_s])[0], np.where(img[:, y_r_s])[0]

    img[:pad, :] = 0
    if len(ind_top) > 0:
        img[pad, ind_top.min():ind_top.max()] = 255

    img[-pad:, :] = 0
    if len(ind_bot) > 0:
        img[-pad, ind_bot.min():ind_bot.max()] = 255

    img[:, -pad:] = 0
    if len(ind_right) > 0:
        img[ind_right.min():ind_right.max(), -pad] = 255

    img[:, :pad] = 0
    if len(ind_left) > 0:
        img[ind_left.min():ind_left.max(), pad] = 255

    return img

def create_valid_zone(imgs, canny_sigma=1):

    pad_with = 10
    valid_zone = np.ones((imgs.shape[1], imgs.shape[2]))
    for i, img in enumerate(imgs[1:], 1):

        img = rgb2gray(img)
        img = np.pad(img, pad_with)
        img_with_edges = canny(img, sigma=canny_sigma).astype(np.uint8)*255
        img_with_edges = canny(gaussian(img_with_edges, sigma=2.5)).astype(np.uint8)*255

        # cv2.imwrite(str(i) + 'countedd.png', img_with_edges)

        curr_valid_zone = find_valid_part(img_with_edges)
        curr_valid_zone = curr_valid_zone[pad_with:-pad_with, pad_with:-pad_with]

        print(' counted :', i, '/', imgs.shape[0]-1)
        # cv2.imwrite(str(i) + 'counted.png', curr_valid_zone.astype(np.uint8)*255)

        valid_zone = np.logical_and(valid_zone, curr_valid_zone)

    return np.where(valid_zone, 255, 0)

# create folder with valid zone for each set of images
def create_valid_zones_with_rotation(ds_folders, exceptions, save_dir):
    
    if not Path(save_dir).exists():
        os.mkdir(save_dir)

    os.chdir(save_dir)
    curr_dir = os.getcwd()

    for dir_name in sorted(Path(ds_folders).glob('*A*'),
                        key = lambda x: str(x).split('/')[-1].split('.')[0]):

        names_in_folder = sorted(list(map(lambda x: str(x), Path(str(dir_name)).glob('*.jpg'))),
                                key = lambda x: int(x.split('/')[-1].split('.')[0]))
        names_in_folder = list(filter(lambda x: not int(x.split('/')[-1].split('.')[0]) in exceptions, names_in_folder))

        imgs_in_folder = list(map(lambda x: cv2.imread(x), names_in_folder))

        num = str(dir_name).split('/')[-1]

        print('Creating valid zone for', num)
        valid_zone = create_valid_zone(np.array(imgs_in_folder))
        print('Saved :', num.split('.')[0] + '_valid_zone.png')

        cv2.imwrite(num.split('.')[0] + '_valid_zone.png', valid_zone.astype(np.uint8))

    os.chdir(curr_dir)

def fill_masks_weights(masks, valid_zones):

    max_index = np.unique(masks).max()
    masks_weights = np.zeros((masks.shape[0], max_index + 1), dtype=np.int64)
    for i, mask in enumerate(masks):

        values, amounts = np.unique(mask[np.where(valid_zones[i])], return_counts=True)
        masks_weights[i, values] = amounts

    masks_weights = masks_weights[:, np.where(masks_weights.sum(axis=0))[0]]

    return masks_weights

# get next patch coords and image(mask) number
def generate_patch_coords(masks, masks_weights, valid_zones, curr_class_stats, patch_size, class_ids, vvalid_zones):

    # not_norm_class_probs = 1 / (curr_class_stats * curr_class_stats)
    # class_probs = not_norm_class_probs / not_norm_class_probs.sum()
    # class_to_find = np.random.choice(np.arange(curr_class_stats.shape[0]), size=1, p=class_probs)[0]
    class_to_find = np.argmin(curr_class_stats)
    
    weights_values = masks_weights[:, class_to_find]
    # weights_values = weights_values * weights_values
    # weights_values = 1 / np.where(weights_values > 0, weights_values, -1)
    # not_norm_weight_probs = 1 / (weights_values * weights_values)
    # weights_values = np.where(weights_values > 0, weights_values, 0)
    p_values = weights_values / weights_values.sum()
    
    img_num_to_cut_from = np.random.choice(np.arange(masks.shape[0]), size=1, p=p_values)[0]
    mask = masks[img_num_to_cut_from]
    mask_01 = np.where(mask == class_ids[class_to_find], 1, 0)

    prob_map = calc_prob_map(mask_01, valid_zones[img_num_to_cut_from], patch_size)

    # print('stats', curr_class_stats[0])
    # print(img_num_to_cut_from, p_values[img_num_to_cut_from], p_values.max(), class_to_find, class_ids[class_to_find], mask_01.sum())
    # print(p_values)
    # print('mask', np.unique(mask[np.where(vvalid_zones[img_num_to_cut_from])], return_counts=True))
    # print(weights_values)

    # print('mask', np.unique(mask, return_counts=True))
    num_in_flattened = np.random.choice(np.arange(mask.shape[0] * mask.shape[1]), size=1, p=prob_map.flatten())[0]
    y, x = int(num_in_flattened / mask.shape[1]), int(num_in_flattened % mask.shape[1])

    # print((x,y), prob_map.flatten()[num_in_flattened], prob_map.max())
    # print('patch', np.unique(mask[y:y+patch_size[0], x:x+patch_size[1]], return_counts=True))
    # mask_01 *= 255
    # mask_01[y:y+patch_size[0], x] = 125
    # mask_01[y, x:x+patch_size[0]] = 125
    # imsave(str(curr_class_stats[0]) + 'prob_01.png', np.concatenate((mask_01, prob_map / prob_map.max() * 255), axis=1))

    patch = mask[y:y+patch_size[0], x:x+patch_size[1]]
    classes, amounts = np.unique(patch, return_counts=True)
    
    for class_id, am in zip(classes, amounts):
        # print('adding', class_id, np.where(class_ids == class_id)[0][0], am)
        curr_class_stats[np.where(class_ids == class_id)[0][0]] += am

    # print("Stats")
    # print(class_to_find, img_num_to_cut_from)
    # print(p_values)

    return (y, x), img_num_to_cut_from, curr_class_stats

# create patch coords to cut from balanced patches from dataset
def create_balanced_dataset_patch_coords(masks, valid_zones, amount_of_patches, patch_size):

    # if not Path(save_dir).exists():
    #     os.mkdir(save_dir)

    # curr_dir = os.getcwd()
    # os.chdir(save_dir)

    corrected_masks = masks * (valid_zones / 255).astype(np.uint8)
    masks_weights = fill_masks_weights(masks, valid_zones)
    class_ids = np.unique(corrected_masks)
    # print('Classes: ', np.unique(corrected_masks, return_counts=True))

    vvalid_zones = valid_zones.copy()
    correct_valid_zones(valid_zones, patch_size)

    prepared_patches = 0
    patchs_coords = [[] for _ in range(masks.shape[0])]
    amounts_ = [0 for _ in range(masks.shape[0])]
    curr_class_stats = np.ones(len(class_ids))
    # print(class_ids, len(class_ids))

    while prepared_patches < amount_of_patches:
        patch_coords, i, curr_class_stats = generate_patch_coords(masks, masks_weights, valid_zones,
                                                        curr_class_stats, patch_size, class_ids, vvalid_zones)

        # print('masks', np.unique(masks, return_counts=True))
        print('Selected: ' + str(prepared_patches + 1) + '/' + str(amount_of_patches))
        print(curr_class_stats / curr_class_stats.sum())
        # print()

        patchs_coords[i].append(patch_coords)
        amounts_[i] += 1
        print('Amounts:')
        print(amounts_)
        prepared_patches += 1
        
    return patchs_coords

def correct_valid_line(line, patch_size):

    valid_row_coords = np.where(line)[0]
    valid_amount = len(valid_row_coords)

    if valid_amount < patch_size[0]:
        line[:] = False

    else:
        tmp = np.roll(valid_row_coords, -1) - valid_row_coords
        tmp[-1] = 2
        end_indeces = valid_row_coords[np.where(tmp > 1)]

        for last_ind in end_indeces:
            if last_ind >= patch_size[0] - 1:
                line[last_ind - patch_size[0] + 1:last_ind + 1] = False
    
def correct_valid_zones(valid_zones, patch_size):

    print('Correcting valid zones accroding to patch size')

    for i, valid_zone in enumerate(valid_zones):
        valid_zone = (valid_zone).astype(np.bool)
        coords = np.where(valid_zone)
        valid_coords_x = np.unique(coords[0])
        valid_coords_y = np.unique(coords[1])

        for row in valid_coords_x:
            correct_valid_line(valid_zone[row], patch_size)

        for col in valid_coords_y:
            correct_valid_line(valid_zone[:, col], patch_size)
        
        valid_zones[i] = valid_zone.astype(np.uint8)
        print(' ' + str(i+1) + '/' + str(valid_zones.shape[0]) + ' corrected')
        # cv2.imwrite('/Users/macbookpro/python_files/python_rep/tmp/' + str(i) +' corrected.png', valid_zones[i] * 255)

def calc_prob_map(mask, valid_zone, patch_size):
    
    # prob_map = np.zeros(valid_zone.shape)
    # for coords in zip(np.where(valid_zone)[0], np.where(valid_zone)[1]):
    #     prob_map[coords[0], coords[1]] = mask[coords[0]:coords[0] + patch_size[0],
    #                                           coords[1]:coords[1] + patch_size[1]].sum()
    basic_sum = np.cumsum(np.cumsum(mask, axis=0), axis=1)
    B = np.roll(np.roll(basic_sum, -patch_size[0], axis=0), -patch_size[1], axis=1)
    C = np.roll(basic_sum, -patch_size[1], axis=0)
    D = np.roll(basic_sum, -patch_size[1], axis=1)
    res = B + basic_sum - C - D
    res[0] = 0
    res[:, 0] = 0
    res = res * valid_zone

    summ = res.sum()
    if summ == 0:
        return res
    else:
        return res / summ

    # prob_map = prob_map * valid_zone
    # return prob_map / prob_map.sum()

# create probability maps in order to use them later in patch sampling
def create_probability_maps(masks, valid_zones, patch_size, save_dir):

    if not Path(save_dir).exists():
        os.mkdir(save_dir)

    curr_dir = os.getcwd()
    os.chdir(save_dir)
    
    all_pixels = patch_size[0] * patch_size[1]
    correct_valid_zones(valid_zones, patch_size)

    amount_of_classes = len(np.unique(masks))
    for class_id in np.unique(masks):
        print("Creating probability maps for class:", class_id)
        for num, mask in enumerate(masks):
            # prob_map = np.zeros(mask.shape, dtype=np.float32)
            print(' Creating the ' + str(num+1) + 'th map')
            mask = np.where(mask == class_id, 1, 0)
            # valid_coords = np.where(valid_zones[num])
            # valid_coords = np.array(np.where(valid_zones[num])).T
            prob_map = calc_prob_map(mask, valid_zones[num], patch_size)
        
            # for coords in valid_coords:
            #     prob_map[coords[0], coords[1]] = mask[coords[0]:coords[0] + patch_size[0],
            #                                             coords[0]:coords[0] + patch_size[0]].sum() / all_pixels

            # prob_map = prob_map / prob_map.max()

            with open(str(class_id) + '.' + str(num) + '.npy', 'wb') as f:
                np.save(f, prob_map)

            print('  Max prob: ', prob_map.max())

    os.chdir(curr_dir)

def create_valid_zones_ordinary(start, end, size, save_dir):

    if not Path(save_dir).exists():
        os.mkdir(save_dir)

    curr_dir = os.getcwd()
    os.chdir(save_dir)

    for i in range(start, end):
        cv2.imwrite(str(i) + '_valid_zone.png', np.ones(size, dtype=np.uint8) * 255)

    os.chdir(curr_dir)

ds_folders = '/Users/macbookpro/python_files/python_rep/rotated_dataset'
masks_folder = '/Users/macbookpro/Downloads/S1_v1/masks/train_mask'
exceptions = [60, 120, 135, 195, 225, 240]
amount_of_patches = 5000#7800
patch_size = (384, 384)
img_size = (2547, 3396)
max_ind = 140
save_dir_valid_zones = '/Users/macbookpro/python_files/python_rep/valid_zones_for_patches' 
# save_dir_dataset = '/Users/macbookpro/python_files/python_rep/balanced_dataset'
# save_dir_probability_maps = '/Users/macbookpro/python_files/python_rep/probability_maps'
excp_num = []#[41, 43, 44, 59]

# create_valid_zones_with_rotation(ds_folders, exceptions, save_dir_valid_zones)
# create_valid_zones_ordinary(61, 140, img_size, save_dir_valid_zones)
# create_valid_zones_ordinary(0, 60, img_size, save_dir_valid_zones)

masks = [cv2.imread(str(name), cv2.IMREAD_GRAYSCALE) for name in sorted(Path(masks_folder).glob('*.png'), 
                                                                    key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
                                                                        if (int(str(name).split('/')[-1].split('.')[0]) < max_ind) and
                                                                            not (int(str(name).split('/')[-1].split('.')[0]) in excp_num)]
valid_zones = [cv2.imread(str(name), cv2.IMREAD_GRAYSCALE) for name in sorted(Path(save_dir_valid_zones).glob('*.png'), 
                                                                        key=lambda x: int(str(x).split('/')[-1].split('_')[0]))
                                                                            if not (int(str(name).split('/')[-1].split('_')[0]) in excp_num) and
                                                                            int(str(name).split('/')[-1].split('_')[0]) < max_ind] 

print(len(masks), len(valid_zones))
patch_coords = create_balanced_dataset_patch_coords(np.array(masks), np.array(valid_zones), amount_of_patches, patch_size)
with open('coords_for_patches.json', 'w') as f:
    json.dump(patch_coords, f)

# print(np.unique(masks, return_counts=True))
# val_zone = cv2.imread("/Users/macbookpro/python_files/python_rep/tmp/37 corrected.png", cv2.IMREAD_GRAYSCALE)
# val_zone = val_zone / val_zone.max()
# mask = masks[37]
# mask_01 = np.where(mask == 11, 1, 0)
# prob_map = calc_prob_map(mask_01, val_zone, patch_size)
# print(prob_map.min(), prob_map.max())
# prob_map = prob_map / prob_map.max()
# cv2.imwrite('/Users/macbookpro/python_files/python_rep/tmp/' + '37_' +' mask.png', mask_01 * 255 * val_zone)
# # prob_map = np.where(mask_01 * 255 * val_zone > 0, 0.5, prob_map)
# cv2.imwrite('/Users/macbookpro/python_files/python_rep/tmp/' + '37__' +' prob.png', prob_map * 255)