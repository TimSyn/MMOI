from pathlib import Path
import numpy as np
import cv2
import tifffile
import os

imgs_path = './patch_samples_plain/img_patches'
masks_path = './patch_samples_plain/mask_patches'
weights_path = './weight_maps_plain'
train_dir = './train_plain'
test_dir = './test_plain'
valid_dir = './validation_plain'
non_pol_amount = 2969
total_amount = 5000
test_size = 0#250
valid_size = 0#400
ratio = 7 / 11

img_names_png = sorted(list(Path(imgs_path).glob('*.png')), key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
img_names_tiff = sorted(list(Path(imgs_path).glob('*.tiff')), key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
img_names = img_names_png + img_names_tiff
mask_names = sorted(list(Path(masks_path).glob('*.png')), key=lambda x: int(str(x).split('/')[-1].split('.')[0]))
weight_names = sorted(list(Path(weights_path).glob('*.npy')), key=lambda x: int(str(x).split('/')[-1].split('.')[0]))

print(len(img_names), len(mask_names), len(weight_names))

non_pol_validation_amount, non_pol_test_amount = int(valid_size * ratio), int(test_size * ratio)
non_pol_nums = np.random.choice(np.arange(0, non_pol_amount), 
                                non_pol_validation_amount,# + non_pol_test_amount, 
                                replace=False)
pol_nums = np.random.choice(np.arange(non_pol_amount, total_amount),
                            valid_size - non_pol_validation_amount,# + test_size - non_pol_test_amount,
                            replace=False)
# nums_valid = np.r_[non_pol_nums[:non_pol_validation_amount], pol_nums[:valid_size - non_pol_validation_amount]]
# nums_test = np.r_[non_pol_nums[non_pol_validation_amount:], pol_nums[valid_size - non_pol_validation_amount:]]
nums = np.r_[non_pol_nums, pol_nums]

if not Path(train_dir).exists():
    os.mkdir(train_dir)
    os.mkdir(train_dir + '/img_patches')
    os.mkdir(train_dir + '/mask_patches')
    os.mkdir(train_dir + '/weight_maps')
if not Path(valid_dir).exists():
    os.mkdir(valid_dir)
    os.mkdir(valid_dir + '/img_patches')
    os.mkdir(valid_dir + '/mask_patches')
    os.mkdir(valid_dir + '/weight_maps')
# if not Path(test_dir).exists():
#     os.mkdir(test_dir)
#     os.mkdir(test_dir + '/img_patches')
#     os.mkdir(test_dir + '/mask_patches')
#     os.mkdir(test_dir + '/weight_maps')

stats = np.zeros((13,))
for img_name, mask_name, weight_name in zip(img_names, mask_names, weight_names):

    img_name = str(img_name)
    num = int(img_name.split('/')[-1].split('.')[0])
    print(num)

    with open(weight_name, 'rb') as f:
        weight_map = np.load(f, allow_pickle=True)

    # if len(np.where((nums_test - num) == 0)[0]) == 1:
    #     save_dir = test_dir
    # elif len(np.where((nums_valid - num) == 0)[0]) == 1:
    #     save_dir = valid_dir
    # else:
    #     save_dir = train_dir

    if len(np.where((nums - num) == 0)[0]) == 1:
        save_dir = valid_dir
    else:
        save_dir = train_dir

    if img_name.split('/')[-1].split('.')[1] == 'png':
        img = cv2.imread(img_name)
        cv2.imwrite(save_dir + '/img_patches/' + str(num) + '.png', img)
    else:
        img = tifffile.imread(img_name)
        tifffile.imsave(save_dir + '/img_patches/' + str(num) + '.tiff', img)

    # counting stats
    # print(stats / stats.sum())
    # ids, counts = np.unique(mask, return_counts=True)
    # for i, id_ in enumerate(ids):
    #     stats[id_] += counts[i]

    mask = cv2.imread(str(mask_name), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(save_dir + '/mask_patches/' + str(num) + '.png', mask)

    with open(save_dir + '/weight_maps/' + str(num) + '.npy', 'wb') as f:
        np.save(f, weight_map)
