import numpy as np
import cv2
import tifffile
from pathlib import Path

def augment_patch(img, augment_arg, is_png=True):

    if 0 <= augment_arg <= 3:
        return rotate_90n_clockwise(img, augment_arg, is_png=is_png)
    if augment_arg == 4:
        return reflect(img, is_vertical=True, is_png=is_png)
    if augment_arg == 5:
        return reflect(img, is_vertical=False, is_png=is_png)

def rotate_90n_clockwise(img, n, is_png=True):

    tmp = img.copy()

    if not is_png:
        for i in range(img.shape[-1]):
            tmp[:, :, i] = rotate_90n_clockwise(img[:, :, i], n)
        
        return tmp

    if n % 4 == 0:
        return img
    if n % 4 == 1:
        for i in range(img.shape[0]):
            tmp[:,-i] = img[i,:]
        return tmp
    if n % 4 == 2:
        return rotate_90n_clockwise(rotate_90n_clockwise(img, 1), 1)
    else:
        for i in range(img.shape[0]):
            tmp[:,i] = img[i,:]
        return tmp

def reflect(img, is_vertical=True, is_png=True):

    tmp = img.copy()

    if is_vertical:

        if is_png:
            for i in range(img.shape[0]):
                tmp[:,-i] = img[:,i]
        else:
            for i in range(img.shape[-1]):
                tmp[:, :, i] = reflect(img[:, :, i], is_vertical=is_vertical, is_png=True)

    else:

        if is_png:
            for i in range(img.shape[0]):
                tmp[-i,:] = img[i, :]
        else:
            for i in range(img.shape[-1]):
                tmp[:, :, i] = reflect(img[:, :, i], is_vertical=is_vertical, is_png=True)

    return tmp

# patch_dir = 'train/img_patches'
# mask_dir = 'train/mask_patches'
# weight_dir = 'train/weight_maps'

# angle = 90
# amount_of_rotation = 3

# patch_names = [str(name) for name in Path(patch_dir).glob('*.png')] + [str(name) for name in Path(patch_dir).glob('*.tiff')]
# patch_names = sorted(patch_names, key = lambda x : int(x.split('/')[-1].split('.')[0]))
# mask_names = sorted([str(name) for name in Path(mask_dir).glob('*.png')], key = lambda x : int(x.split('/')[-1].split('.')[0]))
# weight_names = sorted([str(name) for name in Path(weight_dir).glob('*.npy')], key = lambda x : int(x.split('/')[-1].split('.')[0]))

# counter = len(patch_names)
# for img_name, mask_name, weight_name in zip(patch_names, mask_names, weight_names):

#     is_png = img_name.split('/')[-1].split('.')[1] == 'png'
#     if is_png:
#         img = cv2.imread(img_name)
#     else:
#         img = tifffile.imread(img_name)
        

#     mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

#     with open(weight_name, 'rb') as f:
#         weight = np.load(f, allow_pickle=True)

#     if is_png:
#         cv2.imwrite(patch_dir + '/' + str(counter) + '.png', reflect(img, is_vertical=True, is_png=is_png))
#     else:
#         tifffile.imsave(patch_dir + '/' + str(counter) + '.tiff', reflect(img, is_vertical=True, is_png=is_png))
#     cv2.imwrite(mask_dir + '/' + str(counter) + '.png', reflect(mask, is_vertical=True))
#     with open(weight_dir + '/' + str(counter) + '.npy', 'wb') as f:
#             np.save(f, reflect(weight, is_vertical=True))

#     counter += 1

#     if is_png:
#         cv2.imwrite(patch_dir + '/' + str(counter) + '.png', reflect(img, is_vertical=False, is_png=is_png))
#     else:
#         tifffile.imsave(patch_dir + '/' + str(counter) + '.tiff', reflect(img, is_vertical=False, is_png=is_png))
#     cv2.imwrite(mask_dir + '/' + str(counter) + '.png', reflect(mask, is_vertical=False))
#     with open(weight_dir + '/' + str(counter) + '.npy', 'wb') as f:
#             np.save(f, reflect(weight, is_vertical=False))

#     counter += 1

#     for _ in range(amount_of_rotation):

#         if is_png:
#             img = rotate_90n_clockwise(img)

#             cv2.imwrite(patch_dir + '/' + str(counter) + '.png', img)
#         else:
#             # tmp_img = img.copy()
#             for i, sub_img in enumerate(img):
#                 img[i] = rotate_90n_clockwise(sub_img)

#             tifffile.imsave(patch_dir + '/' + str(counter) + '.tiff', img)

#         mask = rotate_90n_clockwise(mask)
#         weight = rotate_90n_clockwise(weight)

#         cv2.imwrite(mask_dir + '/' + str(counter) + '.png', mask)

#         with open(weight_dir + '/' + str(counter) + '.npy', 'wb') as f:
#             np.save(f, weight)
        
#         print(counter)
#         counter += 1






