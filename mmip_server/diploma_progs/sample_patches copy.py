import cv2
import json
from pathlib import Path
import os
import numpy as np
import tifffile

amount_of_pol_imgs = 20
patch_size = (384, 384)

img_dirs_polarized = '/home/t.synovyat/rotated_dataset'
# img_dirs_polarized = '/Users/macbookpro/python_files/python_rep/rotated_dataset'
masks_dir = '/home/t.synovyat/train_mask_plain'
# masks_dir = '/home/t.synovyat/masks_machine'
# masks_dir = '/Users/macbookpro/Downloads/OneDrive_1_10-8-2020/Box3-5_DS4/masks_machine'
img_dir_non_polarized = '/home/t.synovyat/train_img_plain'
# img_dir_non_polarized = '/home/t.synovyat/img'
# img_dir_non_polarized = '/Users/macbookpro/Downloads/OneDrive_1_10-8-2020/Box3-5_DS4/img'
dataset_dir = 'patch_samples_plain'
img_patch_dir = 'img_patches'
mask_patch_dir = 'mask_patches'
excp_num = []#[41, 43, 44, 59]
pol_from = 40
pol_to = 61
exceptions = [60, 120, 135, 195, 225, 240] # due to bad alignment
counter = 0

if not Path(dataset_dir).exists():
    os.mkdir(dataset_dir)

curr_dir = os.getcwd()
os.chdir(dataset_dir)

if not Path(img_patch_dir).exists():
    os.mkdir(img_patch_dir)

if not Path(mask_patch_dir).exists():
    os.mkdir(mask_patch_dir)

with open('/home/t.synovyat/coords_for_patches.json', 'r') as f:
# with open('/Users/macbookpro/python_files/python_rep/coords_for_patches_old.json', 'r') as f:
    coords = json.load(f)

print(len(coords))
# non polarized
masks_b = [cv2.imread(str(name), cv2.IMREAD_GRAYSCALE) for name in sorted(Path(masks_dir).glob('*.png'), 
                                                                        key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
                                                                            if int(str(name).split('/')[-1].split('.')[0]) < pol_from + 1]

# imgs_non_polarized_b = [cv2.imread(str(name)) for name in sorted(Path(img_dir_non_polarized).glob('*.jpg'),
#                                                                key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
#                                                             if int(str(name).split('/')[-1].split('.')[0]) < pol_from + 1]
# print(len(masks_b))                        
# for img_id in range(0, len(masks_b)):
    
#     if len(coords[img_id]) > 0:
#         img, mask = imgs_non_polarized_b[img_id], masks_b[img_id]
#         for x, y in coords[img_id]:
            
#             img_patch = img[x : x + patch_size[0], y : y + patch_size[1]]
#             mask_patch = mask[x : x + patch_size[0], y : y + patch_size[1]]

#             cv2.imwrite('./' + img_patch_dir + '/' + str(counter) + '.png', img_patch)
#             cv2.imwrite('./' + mask_patch_dir + '/' + str(counter) + '.png', mask_patch)

#             counter += 1
#             print('Sampled :', counter)

# masks_a = [cv2.imread(str(name), cv2.IMREAD_GRAYSCALE) for name in sorted(Path(masks_dir).glob('*.png'), 
#                                                                         key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
#                                                                             if int(str(name).split('/')[-1].split('.')[0]) > pol_to - 1]

# imgs_non_polarized_a = [cv2.imread(str(name)) for name in sorted(Path(img_dir_non_polarized).glob('*.jpg'),
#                                                                key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
#                                                             if int(str(name).split('/')[-1].split('.')[0]) > pol_to - 1]

# print(len(masks_b), len(masks_a), len(imgs_non_polarized_a))
# for img_id in range(len(masks_a)):
    
#     if len(coords[len(masks_b) + 18 + img_id]) > 0:
#         img, mask = imgs_non_polarized_a[img_id], masks_a[img_id]
#         for x, y in coords[len(masks_b) + 18 + img_id]:
            
#             img_patch = img[x : x + patch_size[0], y : y + patch_size[1]]
#             mask_patch = mask[x : x + patch_size[0], y : y + patch_size[1]]

#             cv2.imwrite('./' + img_patch_dir + '/' + str(counter) + '.png', img_patch)
#             cv2.imwrite('./' + mask_patch_dir + '/' + str(counter) + '.png', mask_patch)

#             counter += 1
#             print('Sampled :', counter)

# polarized
masks = [cv2.imread(str(name), cv2.IMREAD_GRAYSCALE) for name in sorted(Path(masks_dir).glob('*.png'), 
                                                                        key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
                                                                            if ((int(str(name).split('/')[-1].split('.')[0]) > pol_from) and (int(str(name).split('/')[-1].split('.')[0]) < pol_to)) and
                                                                                not (int(str(name).split('/')[-1].split('.')[0]) in excp_num)]
img_polarized = [cv2.cvtColor(cv2.imread(str(name)), cv2.COLOR_BGR2RGB) for name in sorted(Path(img_dir_non_polarized).glob('*.jpg'),
                                                                        key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
                                                                            if ((int(str(name).split('/')[-1].split('.')[0]) > pol_from) and (int(str(name).split('/')[-1].split('.')[0]) < pol_to)) and
                                                                                not (int(str(name).split('/')[-1].split('.')[0]) in excp_num)]
# print(len(masks))
# print(np.unique(np.array(masks), return_counts=True))
# for i in range(len(masks)):
#     print(np.unique(np.array(masks[i]), return_counts=True))

counter = 2969
dirs_names = sorted(Path(img_dirs_polarized).glob('*A*'), key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
length_non_pol_b = len(masks_b)
for img_id in range(length_non_pol_b, length_non_pol_b + 18):

    if len(coords[img_id]) > 0:
        names_in_folder = sorted(list(map(lambda x: str(x), Path(str(dirs_names[img_id - length_non_pol_b])).glob('*.jpg'))),
                                key = lambda x: int(x.split('/')[-1].split('.')[0]))
        names_in_folder = list(filter(lambda x: not int(x.split('/')[-1].split('.')[0]) in exceptions, names_in_folder))
        imgs_in_folder = [img_polarized[img_id - length_non_pol_b][np.newaxis, :]] +\
            list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)[np.newaxis, :], names_in_folder))[:-1] #cuz we add 1 more
        img, mask = np.concatenate(imgs_in_folder), masks[img_id - length_non_pol_b]

        for x, y in coords[img_id]:
            
            img_patch = img[:, x : x + patch_size[0], y : y + patch_size[1], :]
            mask_patch = mask[x : x + patch_size[0], y : y + patch_size[1]]

            tifffile.imsave('./' + img_patch_dir + '/' + str(counter) + '.tiff', img_patch)
            cv2.imwrite('./' + mask_patch_dir + '/' + str(counter) + '.png', mask_patch)

            counter += 1
            print('Sampled :', counter)

# os.chdir(curr_dir)