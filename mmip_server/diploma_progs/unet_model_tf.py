import numpy as np
import tensorflow as tf
import tifffile
import matplotlib.pyplot as plt
from tensorflow.image import decode_image, resize, ResizeMethod
from pathlib import Path
import os
from skimage.io import imsave, imread
import json
import shutil
import cv2
from skimage.segmentation import find_boundaries
from augmentation import augment_patch


# created according to papper https://arxiv.org/pdf/1505.04597.pdf

iter_num = 1

def hex_char_to_int(char):

        diff = ord(char) - ord('A')
        if diff >= 0:
                return 10 + diff
        else:
                return int(char)

def hex_to_int(hex_str):

        summ = 0
        for i, char in enumerate(hex_str[-1::-1]):
                summ += hex_char_to_int(char) * (16 ** i)
        
        return summ

def convert_hex_str_to_rgb(hex_str):
        
        hex_str = hex_str.upper()
        r = hex_to_int(hex_str[1:3])
        g = hex_to_int(hex_str[3:5])
        b = hex_to_int(hex_str[5:])
        
        color_map = np.array((170, 170, 3))
        
        print(r,g,b)

        return np.array([r, g, b])

# generate array(amount_of_classes, RGB_Color) from meta.json file
def get_color_array(hex_color_list):

        color_array = np.zeros((len(hex_color_list), 3))
        for i, hex_color in enumerate(hex_color_list):
                
                color_array[i] = convert_hex_str_to_rgb(hex_color)
        
        color_map = np.zeros((color_array.shape[0] * 20, 50, 3))
        for color, i in enumerate(range(0, color_map.shape[0], 20)):
                color_map[i:i+20] = color_array[color]

        imsave("color_map.png", color_map)

        return color_array

# the first one is  background color
color_array = get_color_array(["#000000", "#FF0000", "#CBFF00", "#00FF66", "#0065FF", "#CC00FF", "#FF4C4C", "#DBFF4C", "#4CFF93",\
                                "#4C93FF", "#DB4CFF", "#FF9999", "#eaff99"])

# function to create one mask from the output of Unet tensor using probability thrashold
# create mask for one prediction not the whole batch
def create_one_mask_from_tensor(y_pred_proba, color_array, p_threshold=0.5, just_color=False):

        # soft_max = UnetLoss().soft_max(tensor)
        # print(tf.reduce_max(soft_max), tf.reduce_min(soft_max))

        mask = np.zeros((y_pred_proba.shape[0], y_pred_proba.shape[1], 3))

        # creating by using just probability thrashold
        # for batch_num in range(tensor.shape[0]):
        #         for class_id in range(tensor.shape[3]):

        #                 mask[np.where(soft_max[batch_num, :, :, class_id] > p_threshold)] = color_array[class_id]

        # creating by using maximum probability among classes
        # for batch_num in range(tensor.shape[0]):

        if not just_color:
                ind = np.argmax(y_pred_proba.numpy(), axis=-1)
        else:
                ind = y_pred_proba

        for class_id in range(color_array.shape[0]):

                x_ind, y_ind = np.where(ind == class_id)
                mask[x_ind, y_ind] = color_array[class_id]

        return mask

# loss Class for Unet
# save masks for patches
class PlugLoss(tf.keras.losses.Loss):

        def call(self, y_true, y_pred):

                return tf.constant(0.0, dtype=tf.float32)

class UnetLoss(tf.keras.losses.Loss):

        # calculating soft max across all layers
        def soft_max(self, tensor):

                tensor = tensor - tf.math.reduce_max(tensor)
                exp_tensor = tf.cast(tf.math.exp(tensor), tf.float64)
                pixel_sum_through_channels = tf.expand_dims(tf.math.reduce_sum(exp_tensor, axis=-1), axis=-1)

                return exp_tensor / pixel_sum_through_channels

        def call(self, y_true_oh_and_map, y_pred, epsilon=1e-10):
                
                global iter_num, color_array

                y_pred_proba = tf.cast(self.soft_max(y_pred), tf.float32)
                amount_of_classes = tf.shape(y_true_oh_and_map)[-1] - 1 # minus dim for weight_map
                y_true_one_hot = y_true_oh_and_map[:, :, :, :amount_of_classes]
                weight_map = y_true_oh_and_map[:, :, :, amount_of_classes]
                
                before_log_loss = y_pred_proba * y_true_one_hot
                
                indices = tf.where(before_log_loss == 0.)
                updates = tf.ones(tf.shape(indices)[0])
                before_log_loss = tf.tensor_scatter_nd_update(before_log_loss, indices, updates)

                each_pixel_loss = tf.math.log(tf.math.maximum(before_log_loss, epsilon)) * weight_map[:, :, :, tf.newaxis]
                pixel_amount = tf.cast(tf.reduce_prod(tf.shape(y_pred)[1:]), dtype=tf.float32)
                batch_loss = -tf.reduce_sum(each_pixel_loss / pixel_amount, axis=[1,2,3])

                loss_val = tf.reduce_sum(batch_loss / tf.cast(tf.shape(batch_loss)[0], tf.float32))
                print("Loss on batch", loss_val, "Batch Num: ", iter_num)

                with open("debug.txt", 'at') as f:
                        f.write(str(tf.reduce_sum(tf.cast(tf.math.is_nan(y_pred), tf.uint8)).numpy()) + ' ' + str(tf.reduce_sum(tf.cast(tf.math.is_nan(y_pred_proba), tf.uint8)).numpy()) +' ' + str(tf.reduce_sum(tf.cast(tf.math.is_nan(y_true_oh_and_map), tf.uint8)).numpy())+' ' +\
                                str(tf.reduce_sum(tf.cast(tf.math.is_nan(before_log_loss), tf.uint8)).numpy()) + ' ' +str(tf.reduce_sum(tf.cast(tf.math.is_nan(tf.math.maximum(before_log_loss, epsilon)), tf.uint8)).numpy()) +' ' +\
                                str(tf.reduce_sum(tf.cast(tf.math.is_nan(tf.math.log(tf.math.maximum(before_log_loss, epsilon))), tf.uint8)).numpy()) +' ' +\
                                str(tf.reduce_sum(tf.cast(tf.math.is_nan(each_pixel_loss), tf.uint8)).numpy()) + ' ' +str(tf.reduce_sum(tf.cast(tf.math.is_nan(y_pred), tf.uint8)).numpy()) +' ' + str(tf.reduce_sum(tf.cast(tf.math.is_nan(pixel_amount), tf.uint8)).numpy()) +' ' +\
                                str(tf.reduce_sum(tf.cast(tf.math.is_nan(batch_loss), tf.uint8)).numpy()) + '\n')

                dir_name = "results"
                curr_dir_name = os.getcwd()

                if not os.path.isdir(dir_name):
                        os.mkdir(dir_name)
                elif iter_num == 1:
                        shutil.rmtree(dir_name)
                        os.mkdir(dir_name)

                os.chdir(dir_name)

                # saving results
                # for pred_and_true in zip(y_pred_proba, y_true_one_hot):

                #         pred, target = pred_and_true

                #         pred = create_one_mask_from_tensor(pred, color_array)
                #         target = create_one_mask_from_tensor(target, color_array)

                #         res = np.concatenate((pred, target), axis=1)

                #         imsave("prediction" + "_iter_" + str(iter_num) + ".png", res)
                iter_num += batch_loss.shape[0]

                # with open("loss_values.txt", 'at') as f:
                #         f.write(str(loss_val.numpy()) + ',')

                # #
                # # os.chdir("..")

                # # imsave("full_segmentaion" + str(iter_num) + ".png", create_one_mask_from_tensor(y_pred_proba, color_array)[0])
                # #

                os.chdir(curr_dir_name)

                return loss_val#tf.reduce_sum(batch_loss) / batch_loss.shape[0]

class PlugMetric(tf.keras.metrics.Metric):

        def __init__(self, name="plug_metric", **kwargs):

                super(PlugMetric, self).__init__(name=name, **kwargs)

        def update_state(self, y_true_mask, y_pred_proba, sample_weight=None, amount_of_classes=13):

                pass

        def result(self):

                return tf.constant(0.0, dtype=tf.float32)

class AccuracyMetric(tf.keras.metrics.Metric):

        def __init__(self, name="accuracy", **kwargs):

                super(AccuracyMetric, self).__init__(name=name, **kwargs)
                self.acc_val = self.add_weight(name="acc", initializer='zeros')
                self.img_counter = self.add_weight(name='img_counter', initializer='zeros')
                self.counter = 0
                # self.sep_iou = SeparateIoU()

        def soft_max(self, tensor):

                exp_tensor = tf.math.exp(tensor)
                pixel_sum_through_channels = tf.expand_dims(tf.math.reduce_sum(exp_tensor, axis=-1), axis=-1)

                return exp_tensor / pixel_sum_through_channels

        def update_state(self, y_true_mask, y_pred_proba, sample_weight=None, amount_of_classes=13):
                
                global color_array

                y_true_mask = np.argmax(y_true_mask, axis=-1)#
                max_proba_ind = np.argmax(y_pred_proba, axis=-1)
                intersection = np.where(max_proba_ind == y_true_mask)[0].shape[0]

                union = y_pred_proba.shape[1] * y_pred_proba.shape[2]
                self.acc_val.assign(self.acc_val + intersection / union)
                self.img_counter.assign(self.img_counter + y_pred_proba.shape[0])

                # self.sep_iou.update_state(y_true_mask, y_pred_proba)

        def reset_states(self):

                super(AccuracyMetric, self).reset_states()
                # self.sep_iou.reset_states()

        def result(self):

                # print("IoU :", self.iou_val.value().numpy() / self.img_counter.numpy())
                # print("IoU States: ", self.sep_iou.result().numpy())
                return self.acc_val / self.img_counter

class SeparateIoU(tf.keras.metrics.Metric):

        def __init__(self, name="separate_iou", amount_of_classes=13, print_to_file=False, **kwargs):

                super(SeparateIoU, self).__init__(name=name, **kwargs)
                self.iou_val = self.add_weight(name="iou", initializer='zeros')
                self.img_counter = self.add_weight(name='img_counter', initializer='zeros')
                self.counter = 0
                self.print_to_file = print_to_file
                self.amount_of_classes = amount_of_classes
                self.iou_stats = np.zeros(amount_of_classes)
                self.img_stats = np.zeros(amount_of_classes)#tf.Variable(tf.zeros(amount_of_classes))

        def update_state(self, y_true_mask, y_pred_proba, sample_weight=None):
                
                global color_array

                y_true_mask = np.argmax(y_true_mask, axis=-1)#
                amount_of_classes = y_pred_proba.shape[-1]
                max_proba_ind = np.argmax(y_pred_proba, axis=-1)

                tmp = tf.zeros(self.amount_of_classes)
                correct_pixels = max_proba_ind == y_true_mask
                # print("Inter Union: ", end=' ')
                class_count = 0
                for class_id in range(self.amount_of_classes):

                        intersection = np.where(np.logical_and((y_true_mask == class_id), (max_proba_ind == class_id)))[0].shape[0]
                        union = np.where(y_true_mask == class_id)[0].shape[0] + np.where(max_proba_ind == class_id)[0].shape[0] - intersection
                        # print(str(intersection) + ' ' + str(union), end=' ')
                        if np.where(y_true_mask == class_id)[0].shape[0] == 0:
                                continue
                        class_count += 1

                        # amount = 0.
                        # for img_pixels in y_true_mask:
                        #         amount += tf.cast(tf.math.reduce_any(img_pixels == class_id), dtype=tf.float32)

                        # (print((class_id, amount, intersection / union)))

                        # self.iou_stats[class_id].assign(intersection / union + self.iou_stats[class_id])
                        self.iou_stats[class_id] = intersection / union + self.iou_stats[class_id]
                        # self.img_stats[class_id].assign(amount + self.img_stats[class_id])
                        self.img_stats[class_id] = 1 + self.img_stats[class_id]

                # print("CLASS_COUNTER", class_count)
                # print()
                self.img_counter.assign(self.img_counter + y_pred_proba.shape[0])         
        
        def reset_states(self):

                super(SeparateIoU, self).reset_states()
                # self.iou_stats.assign(tf.zeros(self.amount_of_classes))
                # self.img_stats.assign(tf.zeros(self.amount_of_classes))
                self.img_stats = np.zeros(self.amount_of_classes)
                self.iou_stats = np.zeros(self.amount_of_classes)

        def result(self):

                # print("IoU :", self.iou_val.value().numpy() / self.img_counter.numpy())
                # print("Img states: ", self.img_stats)
                # print("IoU states: ", self.iou_stats / np.where(self.img_stats != 0, self.img_stats, 1.))
                iou_per_class = self.iou_stats / np.where(self.img_stats != 0, self.img_stats, 1.)
                
                if self.print_to_file:
                        with open('iou_per_class.txt', 'at') as f:
                                print(iou_per_class, file=f)

                return tf.reduce_sum(iou_per_class) / np.where(self.img_stats)[0].shape[0]
                # return iou_per_class

class IoU_class(tf.keras.metrics.Metric):
        
        def __init__(self, name="iou_", class_id=None, iou_obj=None, amount_of_classes=13, **kwargs):

                super(IoU_class, self).__init__(name=name + str(class_id), **kwargs)
                self.class_id = class_id
                self.iou_obj = iou_obj
                # self.iou_val = self.add_weight(name="iou", initializer='zeros')
                # self.img_counter = self.add_weight(name='img_counter', initializer='zeros')

        def update_state(self, y_true_mask, y_pred_proba, sample_weight=None):

                # y_true_mask = np.argmax(y_true_mask, axis=-1)#
                # amount_of_classes = y_pred_proba.shape[-1]
                # max_proba_ind = np.argmax(y_pred_proba, axis=-1)

                # intersection = np.where(np.logical_and((y_true_mask == self.class_id), (max_proba_ind == self.class_id)))[0].shape[0]
                # union = np.where(y_true_mask == self.class_id)[0].shape[0] + np.where(max_proba_ind == self.class_id)[0].shape[0] - intersection
                
                # if np.where(y_true_mask == self.class_id)[0].shape[0] == 0:
                #         return
                
                # self.iou_val.assign(self.iou_val + intersection / union)
                # self.img_counter.assign(self.img_counter + 1)
                return


        def reset_states(self):

                super(IoU_class, self).reset_states()
        
        def result(self):

                iou_per_class = self.iou_obj.iou_stats / np.where(self.iou_obj.img_stats != 0, self.iou_obj.img_stats, 1.) / np.where(self.iou_obj.img_stats)[0].shape[0]
                return iou_per_class[self.class_id]

class ExpLayer(tf.keras.layers.Layer):

        def __init__(self, name=''):
                super(ExpLayer, self).__init__(name=name)
                # self.num_outputs = num_outputs

        def call(self, input_):
                return tf.math.exp(input_)

# model of Unet itself
class UnetNN(tf.keras.Model):
        
        # dont forget to add dropout
        def conv2d_layer(self, filters, kernel_size=(3,3),
                         padding='same', activation='relu',
                         strides=(1,1)):

                # change order and add ReLU
                seq = tf.keras.Sequential([tf.keras.layers.Conv2D(
                                                filters=filters, kernel_size=kernel_size,
                                                strides=strides, padding=padding,
                                                kernel_regularizer=tf.keras.regularizers.L2(self.L2_const),
                                                kernel_initializer=self.initializer),
                                           tf.keras.layers.ReLU(),
                                           tf.keras.layers.BatchNormalization()])
                
                return seq

        # why do we need kernel_size and strides ???
        def conv2d_transpose_layer(self, filters, kernel_size=(2,2),
                                   padding='valid', activation='relu',
                                   strides=(2,2)): # check params


                seq = tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(
                                                filters=filters, kernel_size=kernel_size,
                                                strides=strides, padding=padding,
                                                kernel_regularizer=tf.keras.regularizers.L2(self.L2_const),
                                                kernel_initializer=self.initializer),
                                        #   tf.keras.layers.ReLU(),
                                        #   tf.keras.layers.BatchNormalization()
                                        #   tf.keras.layers.Dropout(self.drop_prob)]
                ])
                
                return seq

        def conv1d_layer(self, filters, kernel_size=(1,1), padding='same'):

                return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                              padding=padding, kernel_initializer=self.initializer,
                                              kernel_regularizer=tf.keras.regularizers.L2(self.L2_const))
        
        # changed constant filters to min_channels_in_conv to make flexible !!!
        def __init__(self, in_classes, in_channels=3, n_classes=2, n_filters=64,
                     min_channels_in_conv=64, L2_const=0.00001, drop_prob=0.3, epsilon=1e-15):
        
                super(UnetNN, self).__init__()

                # cropping border value to crop images when concatinating
                # self.cropping_border = [(568 - 392) // 2, (280 - 200) // 2, (136 - 104) // 2, (64 - 56) // 2]

                # instance for preprocessing of image
                self.amount_of_classes = in_classes
                self.L2_const = L2_const
                self.epsilon = epsilon
                self.initializer = tf.keras.initializers.GlorotUniform()
                self.drop_prob = drop_prob
                self.generator_obj = GenerateDatasetUnet(amount_of_classes=in_classes)

                # encoding part
                N_min = min_channels_in_conv * 9

                self.conv2d_layer_1_step_1_down = self.conv2d_layer(filters=n_filters)
                self.conv2d_layer_2_step_1_down = self.conv2d_layer(filters=n_filters)
                self.conv1d_layer_3_step_1_down = self.conv1d_layer(filters=n_filters)
                self.maxpool_step_1_down = tf.keras.layers.MaxPool2D(pool_size=(2,2))

                self.conv2d_layer_1_step_2_down = self.conv2d_layer(filters=n_filters*2)
                self.conv2d_layer_2_step_2_down = self.conv2d_layer(filters=n_filters*2)
                self.conv1d_layer_3_step_2_down = self.conv1d_layer(filters=n_filters*2)
                self.maxpool_step_2_down = tf.keras.layers.MaxPool2D(pool_size=(2,2))

                self.conv2d_layer_1_step_3_down = self.conv2d_layer(filters=n_filters*4)
                self.conv2d_layer_2_step_3_down = self.conv2d_layer(filters=n_filters*4)
                self.conv1d_layer_3_step_3_down = self.conv1d_layer(filters=n_filters*4)
                self.maxpool_step_3_down = tf.keras.layers.MaxPool2D(pool_size=(2,2))

                self.conv2d_layer_1_step_4_down = self.conv2d_layer(filters=n_filters*8)
                self.conv2d_layer_2_step_4_down = self.conv2d_layer(filters=n_filters*8)
                self.conv1d_layer_3_step_4_down = self.conv1d_layer(filters=n_filters*8)
                self.maxpool_step_4_down = tf.keras.layers.MaxPool2D(pool_size=(2,2))

                # bottleneck
                self.conv2d_layer_1_step_bottleneck = self.conv2d_layer(filters=n_filters*16)
                self.conv2d_layer_2_step_bottleneck = self.conv2d_layer(filters=n_filters*16)
                self.conv1d_layer_3_step_bottleneck = self.conv1d_layer(filters=n_filters*16)
                self.conv2d_transpose_layer_3_step_bottleneck = self.conv2d_transpose_layer(filters=n_filters*8)

                # decoding part
                self.conv2d_layer_1_step_1_up = self.conv2d_layer(filters=n_filters*8)
                self.conv2d_layer_2_step_1_up = self.conv2d_layer(filters=n_filters*8)
                self.conv1d_layer_3_step_1_up = self.conv1d_layer(filters=n_filters*8)
                self.conv2d_transpose_layer_3_step_1_up = self.conv2d_transpose_layer(filters=n_filters*4)

                self.conv2d_layer_1_step_2_up = self.conv2d_layer(filters=n_filters*4)
                self.conv2d_layer_2_step_2_up = self.conv2d_layer(filters=n_filters*4)
                self.conv1d_layer_3_step_2_up = self.conv1d_layer(filters=n_filters*4)
                self.conv2d_transpose_layer_3_step_2_up = self.conv2d_transpose_layer(filters=n_filters*2)

                self.conv2d_layer_1_step_3_up = self.conv2d_layer(filters=n_filters*2)
                self.conv2d_layer_2_step_3_up = self.conv2d_layer(filters=n_filters*2)
                self.conv1d_layer_3_step_3_up = self.conv1d_layer(filters=n_filters*2)
                self.conv2d_transpose_layer_3_step_3_up = self.conv2d_transpose_layer(filters=n_filters)

                self.conv2d_layer_1_step_4_up = self.conv2d_layer(filters=n_filters)
                self.conv2d_layer_2_step_4_up = self.conv2d_layer(filters=n_filters)
                self.conv1d_layer_3_step_4_up = self.conv1d_layer(filters=n_filters)

                # should we apply batch norm and regularization here?
                self.conv2d_classifier_layer = tf.keras.layers.Conv2D(
                                                filters=in_classes, kernel_size=(1, 1),
                                                strides=(1,1), padding='same',
                                                kernel_initializer=self.initializer,
                                                kernel_regularizer=tf.keras.regularizers.L2(self.L2_const))
                self.softmax_layer = tf.keras.layers.Softmax(name='softmax')
                self.multiplication_layer = tf.keras.layers.Multiply()
                # self.exp_layer = tf.keras.activations.exponential

        def _set_training_val(self, training):

                self.conv2d_layer_1_step_1_down.layers[2].training=training
                self.conv2d_layer_2_step_1_down.layers[2].training=training
                # self.conv2d_layer_1_step_1_down.layers[3].training=training
                # self.conv2d_layer_2_step_1_down.layers[3].training=training

                self.conv2d_layer_1_step_2_down.layers[2].training=training
                self.conv2d_layer_2_step_2_down.layers[2].training=training
                # self.conv2d_layer_1_step_2_down.layers[3].training=training
                # self.conv2d_layer_2_step_2_down.layers[3].training=training

                self.conv2d_layer_1_step_3_down.layers[2].training=training
                self.conv2d_layer_2_step_3_down.layers[2].training=training
                # self.conv2d_layer_1_step_3_down.layers[3].training=training
                # self.conv2d_layer_2_step_3_down.layers[3].training=training

                self.conv2d_layer_1_step_4_down.layers[2].training=training
                self.conv2d_layer_2_step_4_down.layers[2].training=training
                # self.conv2d_layer_1_step_4_down.layers[3].training=training
                # self.conv2d_layer_2_step_4_down.layers[3].training=training

                self.conv2d_layer_1_step_bottleneck.layers[2].training=training
                self.conv2d_layer_2_step_bottleneck.layers[2].training=training
                # self.conv2d_transpose_layer_3_step_bottleneck.layers[1].training=training
                # self.conv2d_layer_1_step_bottleneck.layers[3].training=training
                # self.conv2d_layer_2_step_bottleneck.layers[3].training=training
                # self.conv2d_transpose_layer_3_step_bottleneck.layers[3].training=training

                self.conv2d_layer_1_step_1_up.layers[2].training=training
                self.conv2d_layer_2_step_1_up.layers[2].training=training
                # self.conv2d_transpose_layer_3_step_1_up.layers[1].training=training
                # self.conv2d_layer_1_step_1_up.layers[3].training=training
                # self.conv2d_layer_2_step_1_up.layers[3].training=training
                # self.conv2d_transpose_layer_3_step_1_up.layers[3].training=training

                self.conv2d_layer_1_step_2_up.layers[2].training=training
                self.conv2d_layer_2_step_2_up.layers[2].training=training
                # self.conv2d_transpose_layer_3_step_2_up.layers[1].training=training
                # self.conv2d_layer_1_step_2_up.layers[3].training=training
                # self.conv2d_layer_2_step_2_up.layers[3].training=training
                # self.conv2d_transpose_layer_3_step_2_up.layers[3].training=training

                self.conv2d_layer_1_step_3_up.layers[2].training=training
                self.conv2d_layer_2_step_3_up.layers[2].training=training
                # self.conv2d_transpose_layer_3_step_3_up.layers[1].training=training
                # self.conv2d_layer_1_step_3_up.layers[3].training=training
                # self.conv2d_layer_2_step_3_up.layers[3].training=training
                # self.conv2d_transpose_layer_3_step_3_up.layers[3].training=training

                self.conv2d_layer_1_step_4_up.layers[2].training=training
                self.conv2d_layer_2_step_4_up.layers[2].training=training
                # self.conv2d_layer_1_step_4_up.layers[3].training=training
                # self.conv2d_layer_2_step_4_up.layers[3].training=training

        def call(self, inputs, training=True):
                
                # assuming value.shape = (batch, height, width, channels)
                # len(inputs)
                # encoding pass
                img = inputs[0]
                weight_map = inputs[1]

                # print(tf.reduce_max(img), tf.reduce_min(img), tf.shape(img))
                # print(tf.reduce_max(weight_map), tf.reduce_min(weight_map), tf.shape(weight_map))

                self._set_training_val(training)

                res = self.conv1d_layer_3_step_1_down(img)
                conv = self.conv2d_layer_1_step_1_down(img)
                conv_res_1 = self.conv2d_layer_2_step_1_down(conv)
                conv_res_1 = conv_res_1 + res
                conv = self.maxpool_step_1_down(conv_res_1)

                res = self.conv1d_layer_3_step_2_down(conv)
                conv = self.conv2d_layer_1_step_2_down(conv)
                conv_res_2 = self.conv2d_layer_2_step_2_down(conv)
                conv_res_2 = conv_res_2 + res
                conv = self.maxpool_step_2_down(conv_res_2)

                res = self.conv1d_layer_3_step_3_down(conv)
                conv = self.conv2d_layer_1_step_3_down(conv)
                conv_res_3 = self.conv2d_layer_2_step_3_down(conv)
                conv_res_3 = conv_res_3 + res
                conv = self.maxpool_step_3_down(conv_res_3)

                res = self.conv1d_layer_3_step_4_down(conv)
                conv = self.conv2d_layer_1_step_4_down(conv)
                conv_res_4 = self.conv2d_layer_2_step_4_down(conv)
                conv_res_4 = conv_res_4 + res
                conv = self.maxpool_step_4_down(conv_res_4)

                # bottleneck
                res = self.conv1d_layer_3_step_bottleneck(conv)
                conv = self.conv2d_layer_1_step_bottleneck(conv)
                conv = self.conv2d_layer_2_step_bottleneck(conv)
                conv = conv + res
                conv = self.conv2d_transpose_layer_3_step_bottleneck(conv)

                # decoding pass
                
                # cropping feature maps
                # crop = self.cropping_border[-1]
                # conv_res_4 = conv_res_4[:, crop:-crop, crop:-crop, :]
                # adding previous feature maps
                conv = tf.keras.layers.concatenate([conv_res_4, conv], axis=-1)
                res = self.conv1d_layer_3_step_1_up(conv)
                conv = self.conv2d_layer_1_step_1_up(conv)
                conv = self.conv2d_layer_2_step_1_up(conv)
                conv = conv + res
                conv = self.conv2d_transpose_layer_3_step_1_up(conv)

                # cropping feature maps
                # crop = self.cropping_border[-2]
                # conv_res_3 = conv_res_3[:, crop:-crop, crop:-crop, :]
                # adding previous feature maps
                conv = tf.keras.layers.concatenate([conv_res_3, conv], axis=-1)
                res = self.conv1d_layer_3_step_2_up(conv)
                conv = self.conv2d_layer_1_step_2_up(conv)
                conv = self.conv2d_layer_2_step_2_up(conv)
                conv = conv + res
                conv = self.conv2d_transpose_layer_3_step_2_up(conv)

                # cropping feature maps
                # crop = self.cropping_border[-3]
                # conv_res_2 = conv_res_2[:, crop:-crop, crop:-crop, :]
                # adding previous feature maps
                conv = tf.keras.layers.concatenate([conv_res_2, conv], axis=-1)
                res = self.conv1d_layer_3_step_3_up(conv)
                conv = self.conv2d_layer_1_step_3_up(conv)
                conv = self.conv2d_layer_2_step_3_up(conv)
                conv = conv + res
                conv = self.conv2d_transpose_layer_3_step_3_up(conv)

                # cropping feature maps
                # crop = self.cropping_border[-4]
                # conv_res_1 = conv_res_1[:, crop:-crop, crop:-crop, :]
                # adding previous feature maps
                conv = tf.keras.layers.concatenate([conv_res_1, conv], axis=-1)
                res = self.conv1d_layer_3_step_4_up(conv)
                conv = self.conv2d_layer_1_step_4_up(conv)
                conv = self.conv2d_layer_2_step_4_up(conv)
                conv = conv + res

                #predict classes
                pred = self.conv2d_classifier_layer(conv)
                tmp = self.multiplication_layer([pred, weight_map[:, :, :, tf.newaxis]])
                pred_proba = self.softmax_layer(pred)

                # log = tf.math.log(tf.math.maximum(pred_proba, self.epsilon))
                # tmp = self.multiplication_layer([log, weight_map[:, :, :, tf.newaxis]])
                # res = tf.math.maximum(tf.keras.activations.exponential(tmp), self.epsilon)

                # print(tf.reduce_max(res), tf.reduce_min(res), tf.shape(res))
                # print(tf.reduce_max(tmp), tf.reduce_min(tmp), tf.shape(pred_proba))
                # print(tf.reduce_max(log), tf.reduce_min(log), tf.shape(log))
                # print(tf.reduce_max(pred_proba), tf.reduce_min(pred_proba), tf.shape(pred_proba))

                return pred_proba, self.softmax_layer(pred)

        def test(self, batch_size, img_path, mask_path='', patch_shape=(384, 384), save_dir='./tmp', metric=AccuracyMetric()):

                global color_array
                # use batch parameters

                img_names = sorted([str(name) for name in img_path.glob('*.jpg')], key = lambda x: int(x.split('/')[-1].split('.')[0]))
                mask_names = sorted([str(name) for name in mask_path.glob('*.png')], key = lambda x: int(x.split('/')[-1].split('.')[0]))

                print(len(img_names), len(mask_names))

                test_dir_img = '/Users/macbookpro/python_files/python_rep/mmip_server/tmp/test_plain/img_patches'
                test_dir_mask = '/Users/macbookpro/python_files/python_rep/mmip_server/tmp/test_plain/mask_patches'
                counter = 0
                for img_name, mask_name in zip(img_names, mask_names):

                        print("Processing :", img_name)
                        img = cv2.imread(img_name)
                        # img = decode_image(tf.io.read_file(img_name), channels=3, expand_animations=False)
                        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
                        # mask = imread(mask_name, as_gray=True)
                        mask_colored = create_one_mask_from_tensor(mask, color_array, just_color=True)
                        mask_colored = cv2.cvtColor(np.array(mask_colored, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.imwrite("mask.png", mask_colored)

                        patches, patches_amount, pad_values = self.generator_obj.create_patches_for_image(img,
                                                                                                         patch_shape=patch_shape,
                                                                                                         is_crop=False)

                        pad_top, pad_bot = pad_values[0]
                        pad_right, pad_left = pad_values[1]

                        img, _ = self.generator_obj.extend_img_to_hold_full_patches(img, patch_shape=(int(patch_shape[0]/2), int(patch_shape[1]/2)))
                        # print(patches_amount, pad_values, img.shape, np.unique(img[:256, :256] - patches[0]))
                        print("Patches for image created")
                        mask_patches, _, _ = self.generator_obj.create_patches_for_image(mask,
                                                                                      patch_shape=patch_shape,
                                                                                      is_crop=False)
                        print(len(patches), len(mask_patches), pad_values)
                        for patch, mask in zip(patches, mask_patches):
                                
                                # imsave(test_dir_img + '/' + str(counter) + '.png', np.array(patch, dtype=np.uint8))
                                cv2.imwrite(test_dir_img + '/' + str(counter) + '.png', np.array(patch, dtype=np.uint8))
                                cv2.imwrite(test_dir_mask + '/' + str(counter) + '.png', np.array(mask, dtype=np.uint8))
                                # imsave(test_dir_mask + '/' + str(counter) + '.png', np.array(mask, dtype=np.uint8), check_contrast=False)
                                counter += 1

                        print("Patches for mask created")
                        continue

                        # minidataset to feed into CNN
                        map_func = lambda x: (tf.py_function(self.generator_obj.preprocess_img, [x, True], Tout=(tf.float32)),\
                                                tf.ones(x.shape[:2], dtype=tf.float32))
                        patches = tf.data.Dataset.from_tensor_slices(patches).map(map_func).batch(batch_size)

                        final_img = np.ones(img.shape, dtype=np.float32) * (-1)
                        x, y = 0, 0
                        res_counter = 0
                        for batch in patches:

                                predictions_proba = self.call(batch, training=False)[1]
                                # predictions_proba = tf.nn.softmax(predictions, axis=-1)
                                masks = np.array(mask_patches[res_counter:res_counter+batch_size])

                                # metric.update_state(masks, predictions_proba)

                                dy, dx = int(patch_shape[1] / 2), int(patch_shape[0] / 2)
                                for counter in range(predictions_proba.shape[0]):

                                        if (res_counter % patches_amount[1] == 0) and (res_counter != 0):
                                                y += dy
                                                x = 0

                                        # print(x,y, np.unique(final_img[y : y + patch_shape[0], x : x + patch_shape[1], 0]))
                                        assign_patch = final_img[y : y + patch_shape[0], x : x + patch_shape[1], 0]
                                        new_patch_weights = np.where(assign_patch == -1 , 1, 1/2)
                                        old_patch_weights = np.where(assign_patch != -1 , 1/2, 0)
                                        color_pred = create_one_mask_from_tensor(tf.constant(predictions_proba[counter]), color_array)
                                        color_mask = create_one_mask_from_tensor(masks[counter], color_array, just_color=True)
                                        # color_pred = color_mask
                                        final_img[y : y + patch_shape[0], x : x + patch_shape[1]] = final_img[y : y + patch_shape[0], x : x + patch_shape[1]] * old_patch_weights[:, :, np.newaxis] +\
                                                                                new_patch_weights[:, :, np.newaxis] * color_pred
                                        x += dx

                                        # imsave(save_dir + '/' + str(res_counter) + '.png',\
                                        #         np.concatenate((color_pred, color_mask), axis=1))
                                        # imsave(save_dir + '/n' + str(res_counter) + '.png',\
                                        #         new_patch_weights * 255)
                                        # imsave(save_dir + '/o' + str(res_counter) + '.png',\
                                        #         old_patch_weights * 255)      
  
                                        tmp = np.where(final_img == -1, 0, final_img)[pad_top:-pad_bot, pad_left:-pad_right]
                                        imsave(save_dir + '/а' + str(res_counter) + '.png',\
                                                np.concatenate((tmp, mask_colored), axis=1))
                                        imsave(save_dir + '/p' + str(res_counter) + '.png',\
                                                np.concatenate((color_pred, color_mask), axis=1))
                                        print(batch[0].shape, batch[0].dtype, batch[0].dtype, predictions_proba.dtype, batch[1].shape)
                                        # imsave(save_dir + '/p' + str(res_counter) + '.jpg', batch[0][counter][:,:,:3])
                                        # imsave(save_dir + '/m' + str(res_counter) + '.jpg', masks[counter])

                                        metric.update_state(masks[counter], predictions_proba[counter])
                                        print("IoU :", metric.result())

                                        res_counter += 1

                        # print("IoU :", metric.result())
                        
                        final_img = final_img[pad_top:-pad_bot, pad_left:-pad_right]
                        imsave(save_dir + '/res.jpg', np.concatenate((final_img, mask_colored), axis=1))
                        break

                return

        # def evaluate(self, dataset, loss=UnetLoss(), metric=MeanIoUMetric()):

        #         loss_accum = 0
        #         for img, target in dataset:

        #                 img, target = img[tf.newaxis, :], target[tf.newaxis, :]
        #                 res = self.call(img, training=False)
        #                 loss_accum += loss.call(target, res)
        #                 metric.update_state(target, res)
                
        #         return float(loss_accum / len(dataset)), float(metric.result())

# getting two directories
#       the first one for image samples
#       the second one for their masks
class GenerateDatasetUnet():

        # img.shape = img.hieght, img.width, img.channels

        # take_n = 1 for old weight_maps, = 2 for new
        def get_num_from_name(posix_name, take_n=1):

                splited = posix_name.split('/')[-1].split('.')

                res = []
                for i, val in enumerate(splited):
                        if i < take_n:
                                # print(val)
                                # print(posix_name)
                                res.append(int(val))

                return res
        
        def __init__(self, amount_of_classes=0, X_path="", Y_path="", rotations_amount=18,
                        weight_map_path="./weight_maps", resize_value=(384, 384), shuffle=True):

                # X, Y, Weight_map _paths are all Path instances
                self.Y_path = Y_path
                self.rotations_amount = rotations_amount
                self.X_path = X_path
                self.shuffle = shuffle
                self.weight_maps_path = Path(weight_map_path)
                self.amount_of_classes = amount_of_classes
                self.resize_value = tf.convert_to_tensor(resize_value, dtype=tf.int32)

        def get_sorted_full_file_names(self, path, pattern='*', key=None):

                if key == None:
                        sort_func = GenerateDatasetUnet.get_num_from_name
                else:
                        sort_func = key

                return sorted([str(posix_name) for posix_name in Path(path).glob(pattern)],\
                                       key=sort_func, reverse=False)

        # method to generate dataset
        def __call__(self, is_augment=True, identity_weight_maps=False):
                
                self.X_names = self.get_sorted_full_file_names(self.X_path)
                self.Y_names = self.get_sorted_full_file_names(self.Y_path)
                self.weight_map_names = self.get_sorted_full_file_names(self.weight_maps_path, pattern='*.npy')

                tf.print(len(self.X_names), len(self.Y_names), len(self.weight_map_names))

                self.created_amount = 0
                # process = lambda x, y, z: tf.py_function(self.load_and_preprocess_images, [x, y, z],\
                #                                                 Tout=(tf.float32, tf.float32))

                # elem is (image, one_hot_masks)
                # dataset = tf.data.Dataset.from_tensor_slices((self.X_names,\
                #                                                 self.Y_names,\
                #                                                 self.weight_map_names)).map(process)

                # process_sample = lambda x: tf.py_function(self.load_and_preprocess_img, [x], Tout=(tf.float32))
                # process_mask = lambda x: tf.py_function(self.load_mask, [x], Tout=(tf.float32))
                # process_weight = lambda x: tf.py_function(self.load_weights, [x], Tout=(tf.float32))

                # map_func = lambda x, y: ((tf.py_function(self.load_and_preprocess_img, [x[0]], Tout=(tf.float32)), 
                #                          tf.py_function(self.load_weights, [x[1]], Tout=(tf.float32))),\
                #                         tf.py_function(self.load_mask, [y], Tout=(tf.float32)))

                def map_func(x, y):

                        if is_augment:
                                augment_arg = np.random.choice(np.arange(6))
                        else:
                                augment_arg = -1

                        if not identity_weight_maps:
                                return ((tf.py_function(self.load_and_preprocess_img, [x[0], augment_arg], Tout=(tf.float32)), 
                                        tf.py_function(self.load_weight_map, [x[1], augment_arg], Tout=(tf.float32))),\
                                        tf.py_function(self.load_mask, [y, augment_arg], Tout=(tf.float32)))
                        else:
                                return ((tf.py_function(self.load_and_preprocess_img, [x[0], augment_arg], Tout=(tf.float32)), 
                                        tf.ones(self.resize_value)),\
                                        tf.py_function(self.load_mask, [y, augment_arg], Tout=(tf.float32)))


                # map_func = lambda x, y: tf.py_function(self.load_img_weight_mask, [[x1,x2], x3], Tout=[[tf.float32, tf.float32], tf.float32])

                samples = tf.data.Dataset.from_tensor_slices(self.X_names)#.map(process_sample)
                masks = tf.data.Dataset.from_tensor_slices(self.Y_names)#.map(process_mask)
                weights = tf.data.Dataset.from_tensor_slices(self.weight_map_names)#.map(process_weight)

                tf.print("Dataset was created.")

                # return tf.data.Dataset.zip((samples, weights)), masks
                if self.shuffle:
                        return tf.data.Dataset.zip((tf.data.Dataset.zip((samples, weights)), masks)).shuffle(len(self.X_names)).map(map_func)
                else:
                        return tf.data.Dataset.zip((tf.data.Dataset.zip((samples, weights)), masks)).map(map_func)

        # correct everything accordingly to patches sizes !!!
        # preprocessing in order to pass to Unet
        def load_img_weight_mask(self, x, y):

                return (self.load_and_preprocess_img(x[0]), self.load_weight_map(x[1])), self.load_mask(y)

        def preprocess_img(self, img, augment_arg=-1, is_repeat=False):

                # img = img / 255

                # paddings = tf.constant([[pad_value, pad_value],
                #                         [pad_value, pad_value], 
                #                         [0, 0]])
                # img = tf.pad(img, paddings, "REFLECT") # ???

                if is_repeat:
                        img = tf.repeat(img[tf.newaxis, :], self.rotations_amount, axis=0)
                        img = tf.concat([x for x in img], axis=-1)

                img = tf.cast(img, dtype=tf.float32)
                
                if augment_arg != -1:
                        img = augment_patch(img.numpy(), augment_arg, is_png=False)
                # means = tf.reduce_mean(img, axis=[0, 1])
                # corr = img - means[tf.newaxis, tf.newaxis, :]
                # stds = tf.math.sqrt(tf.reduce_sum(corr * corr, axis=[0,1]) / ((img.shape[0] - 1) * (img.shape[1] - 1)))
                # img = corr / stds[tf.newaxis, tf.newaxis, :]

                # print('means', means)
                # print('stds', stds)
                # print('corr', corr.numpy().max(), corr.numpy().min())                
                # print('img', img.numpy().max(), img.numpy().min())                

                return tf.constant(img)

        def load_origin_sizes_one_hot(self, x_path, y_path):

                # loading
                x = decode_image(tf.io.read_file(x_path), channels=3, expand_animations=False)
                y_mask = decode_image(tf.io.read_file(y_path), channels=1, expand_animations=False)

                y_mask = tf.squeeze(y_mask)

                # creating one hot tensor to use it in Unet loss + 1 dimension for weight map
                tmp = tf.zeros((y_mask.shape[0], y_mask.shape[1],\
                        self.amount_of_classes), dtype=tf.float32) # add dimension for weight map
                y_one_hot = tf.Variable(tmp, dtype=tf.float32)

                y_one_hot[:, :, :].assign(self.create_one_hot(y_mask))

                return (x, y_one_hot.read_value())

        # create one-hot tesnor for segmentation mask
        # each dim corresponds to different class
        def create_one_hot(self, y_mask):

                size_value = y_mask.shape[:2]
                tmp = tf.zeros((size_value[0], size_value[1],\
                        self.amount_of_classes), dtype=tf.float32)
                y_one_hot = tf.Variable(tmp, dtype=tf.float32)
                for class_id in tf.range(self.amount_of_classes):

                        class_indices = tf.where(y_mask == tf.cast(class_id, dtype=tf.uint8)) # for 0 class
                        # print("HERE SHAPE", class_indices.shape, class_id.numpy())
                        class_indices = tf.cast(class_indices, dtype=tf.int32)
                        # class_id_indices = tf.zeros((tf.shape(class_indices)[0], 1), dtype=tf.int32) + class_id
                        
                        indices = class_indices
                        updates = tf.gather_nd(y_one_hot[:, :, class_id], indices) + 1
                        y_one_hot[:, :, class_id].assign(tf.tensor_scatter_nd_update(y_one_hot[:, :, class_id],\
                                                        indices, updates))

                        # imsave("sized_mask"+str(class_id.numpy())+".png", y_one_hot[:, :, class_id])

                return y_one_hot.read_value()

        def load_and_preprocess_img(self, x_path, augment_arg=None, non_polarized=False):

                flag = False
                if str(x_path.numpy()).split('.')[-1][:-1] != 'tiff':
                        # x = decode_image(tf.io.read_file(x_path), channels=3, expand_animations=False)
                        x = cv2.imread(str(x_path.numpy())[2:-1])
                        x = self.preprocess_img(x, augment_arg)
                        x = tf.repeat(x[tf.newaxis, :], self.rotations_amount, axis=0)
                else:
                        x = tf.convert_to_tensor(tifffile.imread(str(x_path.numpy())[2:-1])) # plus alpha
                        flag = True
                        # x = tfio.experimental.image.decode_tiff(x_path)
                        if non_polarized:
                                # x = self.preprocess_img(x[0], augment_arg)
                                x = self.preprocess_img(cv2.imread(str(x_path.numpy())[2:-1]), augment_arg)
                                x = tf.repeat(x[tf.newaxis, :], self.rotations_amount, axis=0)
                        #       x = tf.repeat(x[tf.newaxis, :], self.rotations_amount, axis=0)
                
                if flag and not non_polarized:
                        x = tf.concat([cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR) for img in x], axis=-1)
                else:
                        x = tf.concat([img for img in x], axis=-1)

                if flag and not non_polarized:
                        x = self.preprocess_img(x, augment_arg)
                # x = self.preprocess_img(x, augment_arg)

                return x
        
        def load_mask(self, mask_path, augment_arg=-1):

                # mask = decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
                # mask = tf.squeeze(mask)
                mask = tf.convert_to_tensor(cv2.imread(str(mask_path.numpy())[2:-1], cv2.IMREAD_GRAYSCALE))

                if augment_arg != -1:
                        mask = tf.constant(augment_patch(mask.numpy(), augment_arg, is_png=True))

                # return tf.cast(mask, dtype=tf.float32)
                return tf.keras.utils.to_categorical(mask, self.amount_of_classes)
        
        def load_weight_map(self, weight_map_path, augment_arg=-1):

                with open(str(weight_map_path.numpy())[2:-1], 'rb') as f:
                        weight_map = np.load(f, allow_pickle=True)
                # weight_map = tf.squeeze(weight_map)

                if augment_arg != -1:
                        weight_map = tf.constant(augment_patch(weight_map, augment_arg, is_png=True))

                return tf.cast(weight_map, dtype=tf.float32)

        # creating one sample of dataset
        def load_and_preprocess_images(self, x_path, y_path, weight_map_path, non_polarized=True):

                # loading
                # if str(x_path.numpy()).split('.')[-1][:-1] != 'tiff':
                #         x = decode_image(tf.io.read_file(x_path), channels=3, expand_animations=False)
                #         x = tf.repeat(x[tf.newaxis, :], self.rotations_amount, axis=0)
                # else:
                #         x = tf.convert_to_tensor(tifffile.imread(str(x_path.numpy())[2:-1]))
                #         if non_polarized:
                #               x = tf.repeat(x[0][tf.newaxis, :], self.rotations_amount, axis=0)
                
                # x = tf.concat([img for img in x], axis=-1)

                # y_mask = decode_image(tf.io.read_file(y_path), channels=1, expand_animations=False)

                # with open(weight_map_path.numpy(), 'rb') as f:
                #         weight_map = np.load(f, allow_pickle=True)
                
                # preprocessing resizing, ...
                # x = self.preprocess_img(x)
                # y_mask = tf.squeeze(y_mask)
                # weight_map = tf.squeeze(weight_map)

                # creating one hot tensor to use it in Unet loss + 1 dimension for weight map
                tmp = tf.zeros((self.resize_value[0], self.resize_value[1],\
                        self.amount_of_classes + 1), dtype=tf.float32) # add dimension for weight map
                y_one_hot_and_map = tf.Variable(tmp, dtype=tf.float32)

                y_one_hot_and_map[:, :, :-1].assign(self.create_one_hot(y_mask))
                y_one_hot_and_map[:, :, self.amount_of_classes].assign(tf.cast(weight_map, dtype=tf.float32))

                return (x, y_one_hot_and_map.read_value())

        # find boundaries for each class in target image
        def find_boundaries_for_masks(self, class_masks):

                boundaries_X = []
                boundaries_Y = []
                # tf.print(">Start creating boundaries...")
                for class_mask_id in range(class_masks.shape[-1]):

                        boundary = find_boundaries(class_masks[:,:,class_mask_id], mode='inner')
                        X_b, Y_b = np.nonzero(boundary)

                        if X_b.shape[0] == 0:
                                continue
                        else:
                                boundaries_X.append(X_b)
                                boundaries_Y.append(Y_b)
                
                # tf.print("Created boundaries.<")

                return boundaries_X, boundaries_Y

        # create weight map to balance class frequencies
        def create_class_balance_map(self, class_masks, min_val=1, max_val=6):

                
                # class_amount_array = np.zeros(self.amount_of_classes)
                class_amount_array = np.sum(class_masks, axis=(0,1))
                if (np.where(class_amount_array > 0, 1, 0)).sum() == 1:
                        return (np.ones(class_masks.shape[:2])).ravel()
                # for class_id in range(self.amount_of_classes):
                #       class_amount_array[class_id] += np.where(class_masks[:, :, class_id] == 1)[0].shape[0]

                class_prob_array = class_amount_array# / class_amount_array.sum()
                # print(class_prob_array)
                tmp = np.where(class_prob_array == 0, -1, class_prob_array)
                tmp = 1 / tmp
                minimum, maximum = tmp[tmp>0].min(), tmp[tmp>0].max()
                k = (max_val - min_val) / (maximum - minimum)
                b = min_val - k * minimum
                tmp[tmp > 0] = k * tmp[tmp > 0] + b
                assign_values = np.where(tmp < 0, 1, tmp)
                # print(tmp, len(class_prob_array[class_prob_array > 0]))
                # assign_values = np.clip(1 / tmp, min_val, max_val)
                # print(assign_values)
                w_1 = np.zeros(class_masks.shape[:2])
                for class_id in range(self.amount_of_classes):

                        if (class_prob_array[class_id] != 0):

                                x_ind, y_ind = np.where(class_masks[:, :, class_id] == 1)
                                # new_max_val = max_val + max_val * np.exp(-10*(class_prob_array[class_id] - tmp.min()))
                                # val = self._clip(assign_values[class_id], min_val, new_max_val)
                                w_1[x_ind, y_ind] = assign_values[class_id]

                return w_1.ravel()

        def _clip(self, val, _min, _max):

                if _min <= val <= _max:
                        return val
                elif val > _max:
                        return _max
                else:
                        return _min

        def create_weight_map(self, class_masks, w_0=5, sigma=5):

                # boundaries_X, boundaries_Y = self.find_boundaries_for_masks(class_masks)
                dist_maps = []

                X_plain, Y_plain = np.meshgrid(np.arange(self.resize_value[0]), np.arange(self.resize_value[1]))
                X_plain, Y_plain = X_plain.flatten().reshape(1, -1), Y_plain.flatten().reshape(1, -1)
                # for X_b, Y_b in zip(boundaries_X, boundaries_Y):
                for class_id in range(class_masks.shape[2]):

                        boundary = find_boundaries(class_masks[:,:,class_id], mode='inner')
                        X_b, Y_b = np.nonzero(boundary)

                        if X_b.shape[0] == 0:
                                continue

                        X_differences_sq = (X_b.reshape(-1, 1) - X_plain) ** 2
                        Y_differences_sq = (Y_b.reshape(-1, 1) - Y_plain) ** 2
                        dist = np.sqrt(X_differences_sq + Y_differences_sq).min(axis=0).reshape(self.resize_value[0], -1).T
                        dist_maps.append(dist)

                if len(dist_maps) > 0:

                        dist_maps = np.sort(np.array(dist_maps), axis=0)
                        if len(dist_maps) > 1:
                                d1, d2 = dist_maps[0, :, :], dist_maps[1, :, :]
                        else:
                                d1 = dist_maps[0, :, :]
                                d2 = np.zeros(d1.shape)
                                
                        edge_weight = w_0 * np.exp( -1 * (d1.ravel() + d2.ravel())** 2 / (2 * sigma ** 2) )
                else:
                        edge_weight = 0

                w_1 = self.create_class_balance_map(class_masks)
                weight_map = w_1 + edge_weight

                return weight_map.reshape(class_masks.shape[:2]), w_1.reshape(class_masks.shape[:2])

        def create_weight_map_low_memory_usage(self, class_masks, w_0=5, sigma=5):

                # boundaries_X, boundaries_Y = self.find_boundaries_for_masks(class_masks)
                
                # # calculating closest dist to the boundaries for each point
                # # the first dim is the closest dist
                # dist_array = np.zeros((class_masks.shape[0], class_masks.shape[1], 2))
                # can_run = len(boundaries_X) > 0

                # if can_run:
                #         for x in range(class_masks.shape[0]):

                #                 for y in range(class_masks.shape[1]):

                #                         dist_values = []
                #                         for X_b, Y_b in zip(boundaries_X, boundaries_Y):

                #                                 X_difference_sq = (X_b - x) ** 2
                #                                 Y_difference_sq = (Y_b - y) ** 2
                #                                 dist = np.sqrt(X_difference_sq + Y_difference_sq).min(axis=0)
                #                                 dist_values.append(dist)
        
                #                         if len(dist_values) == 1:
                #                                 dist_array[x, y, 0] = dist_values[0]
                #                         else:
                #                                 dist_values = sorted(dist_values)       
                #                                 dist_array[x, y, 0] = dist_values[0]
                #                                 dist_array[x, y, 1] = dist_values[1]   

                # if can_run:
                #         d1, d2 = dist_array[:, :, 0], dist_array[:, :, 1]
                #         edge_weight = w_0 * np.exp( -1 * (d1.ravel() + d2.ravel())** 2 / (2 * sigma ** 2) )
                # else:
                #         edge_weight = 0

                w_1 = self.create_class_balance_map(class_masks)
                weight_map = w_1 #+ edge_weight

                return weight_map.reshape(class_masks.shape[:2]), w_1.reshape(class_masks.shape[:2])

        # create dir in local dir with weight maps
        # adding target label to the weight map
        def mkdir_with_weight_maps(self):
                
                target_names = list(filter(lambda x: int(str(x).split('/')[-1].split('.')[0]) > -1, self.get_sorted_full_file_names(self.Y_path)))
                
                if not self.weight_maps_path.exists():
                        os.mkdir(self.weight_maps_path)
                
                # curr_dir_name = os.getcwd()
                # os.chdir(self.weight_maps_path.name)
                counter = 0
                length = len(list(target_names))

                for name in target_names:
        
                        tf.print(">Creating weight map ...")

                        img = decode_image(tf.io.read_file(name), channels=1, expand_animations=False)
                        img = tf.squeeze(img)

                        y_one_hot = self.create_one_hot(img)
                        # tf.print("Created one hot represetation.")
                        weight_map, w_1 = self.create_weight_map_low_memory_usage(y_one_hot)
                        # weight_map, w_1 = self.create_weight_map(y_one_hot)

                        # print(weight_map.max(), weight_map.min(), w_1.max(), w_1.min())

                        num = str(GenerateDatasetUnet.get_num_from_name(name)[0])
                        # img_num = str(num[0]) if num[0] > 9 else '0' + str(num[0])
                        # patch_num = str(num[1]) if num[1] > 9 else '0' + str(num[1]) 

                        with open(str(self.weight_maps_path) + '/' + num + ".weight_map.npy", 'wb') as f:
                                np.save(f, weight_map)


                        # imsave(str(self.weight_maps_path) + '/' + num + ".weight_map.png", weight_map / weight_map.max())
                        print(weight_map.min(), weight_map.max())
                        # with open("teeest.txt", 'wt') as f:
                        #         np.savetxt(f, weight_map)
                
                        # with open("teeest_1.txt", 'wt') as f:
                        #         np.savetxt(f, w_1)
                        # imsave(str(num) + ".weight_map_w_1.png", w_1)
                        counter += 1
                        print("Created weight map " + str(counter) + "/" + str(length) + " .<")


                # os.chdir(curr_dir_name)

        def crop(self, img, shape):

                height, width = shape[0], shape[1]
                delta_height, delta_width = img.shape[0] - height, img.shape[1] - width

                crop_right = delta_width // 2
                crop_left = delta_width // 2

                crop_top = delta_height // 2
                crop_bot = delta_height // 2

                # if crop_value is not even add 1 row or column accordingly
                if delta_width % 2 != 0:
                        crop_right += 1
                if delta_height % 2 != 0:
                        crop_top += 1

                cropped_image = img[crop_top:-crop_bot, crop_right:-crop_left]

                return cropped_image

        def glue_patches(self, patches, patches_amount):

                patches_along_y, patches_along_x = patches_amount

                lines = []
                for y in range(patches_along_y):

                        start = patches_along_x * y
                        end = patches_along_x * (y + 1)

                        line = patches[start:end]
                        lines.append(np.concatenate(line, axis=1))
                
                return np.concatenate(lines, axis=0)
                                
        # dont crop an image but enxtend it to hold full patches
        #  image.shape = [height, width, channels]
        # adding more than original image ???
        def extend_img_to_hold_full_patches(self, img, patch_shape):

                height, width = img.shape[:2]
                p_height, p_width = patch_shape

                full_patches_height = height / p_height
                full_patches_width = width / p_width

                delta_height = p_height - (height - int(full_patches_height) * p_height)
                delta_width = p_width - (width - int(full_patches_width) * p_width)

                pad_top = delta_height // 2
                pad_bot = pad_top

                pad_right = delta_width // 2
                pad_left = pad_right

                if delta_height % 2 != 0:
                        pad_top += 1

                if delta_width % 2 != 0:
                        pad_right += 1

                if len(img.shape) == 2:
                        pad = np.array([[pad_top, pad_bot],[pad_right, pad_left]])
                elif len(img.shape) == 3:
                        pad = np.array([[pad_top, pad_bot],[pad_right, pad_left], [0,0]])

                return np.pad(img, pad), ((pad_top, pad_bot), (pad_right, pad_left))

        # create patches for particular image (crop_shape must be divided by patch_shape!!!)
        # patches_amount = crop_shape / patch_shape
        # return list[crop_shape[0], crop_shape[1], img.shape[2]] with length patches_amount[0] * patches_amount[1]
        def create_patches_for_image(self, img, crop_shape='', patch_shape='', is_crop=True):

                if is_crop:
                        img, pad_values = self.crop(img, crop_shape), None
                else:
                        img_, pad_values = self.extend_img_to_hold_full_patches(img, (int(patch_shape[0] / 2), int(patch_shape[1] / 2)))
                
                pad_top, pad_bot = pad_values[0]
                pad_right, pad_left = pad_values[1]

                if len(img.shape) == 3:
                        pad_width = [[pad_top, pad_bot], [pad_left, pad_right], [0,0]]
                else:
                        pad_width = [[pad_top, pad_bot], [pad_left, pad_right]]
                img = np.pad(img, pad_width, mode='reflect')

                patches_amount = np.array(img.shape[:2]) / (np.array(patch_shape) / 2)
                patches_amount = int(patches_amount[0]) - 1, int(patches_amount[1]) - 1 # patches not 128x128 but 256x256 with stride 1
                x_shape, y_shape = patch_shape
                patches = []
                if len(img.shape) == 3:
                        pad_with = [[0,0], [0,0], [0,0]]
                else:
                        pad_with = [[0,0], [0,0]]

                for raw in range(int(patches_amount[0])):

                        top_ind = int(raw * y_shape / 2)
                        bot_ind = int(top_ind + y_shape / 2)

                        for col in range(int(patches_amount[1])):

                                right_ind = int(col * x_shape / 2)
                                left_ind = int(right_ind + x_shape / 2)

                                # if pad_values:
                                #         if raw == 0:
                                #                 if col == 0:
                                #                         img_tmp = img[pad_top:y_shape, pad_right:x_shape]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = pad_top, 0, pad_right, 0
                                #                         print(pad_with, img_tmp)
                                #                         imsave("tmpp.jpg", img_tmp)
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')
                                #                 elif col == int(patches_amount[1]) - 1:
                                #                         img_tmp = img[pad_top:y_shape, right_ind:pad_left]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = pad_top, 0, 0, pad_left
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')
                                #                 else:
                                #                         img_tmp = img[pad_top:y_shape, right_ind:left_ind]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = pad_top, 0, 0, 0
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')

                                #         elif raw == int(patches_amount[0]) - 1:
                                #                 if col == 0:
                                #                         img_tmp = img[top_ind:pad_bot, pad_right:x_shape]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = 0, pad_bot, pad_right, 0
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')
                                #                 elif col == int(patches_amount[1]) - 1:
                                #                         img_tmp = img[top_ind:pad_bot, right_ind:pad_left]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = 0, pad_bot, 0, pad_left
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')
                                #                 else:
                                #                         img_tmp = img[top_ind:pad_bot, right_ind:left_ind]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = 0, pad_bot, 0, 0
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')

                                #         else:
                                #                 if col == 0:
                                #                         img_tmp = img[top_ind:bot_ind, pad_right:x_shape]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = 0, 0, pad_right, 0
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')
                                #                 elif col == int(patches_amount[1]) - 1:
                                #                         img_tmp = img[top_ind:bot_ind, right_ind:pad_left]
                                #                         pad_with[0][0], pad_with[0][1],\
                                #                         pad_with[1][0], pad_with[1][1] = 0, 0, 0, pad_left
                                #                         patch = np.pad(img_tmp, pad_with, mode='reflect')
                                #                 else:
                                #                         patch = img[top_ind:pad_bot, right_ind:left_ind]
                                # else:
                                #         patch = img[top_ind:bot_ind, right_ind:left_ind].copy()

                                patch = img[top_ind:top_ind+patch_shape[0], right_ind:right_ind+patch_shape[1]].copy()
                                patches.append(patch)
                
                return patches, patches_amount, pad_values

        # create patches for images and masks in speciefied directories in contructor
        def create_patches_for_directories(self, crop_shape=(388*6, 388*8), patch_shape=(384, 384),\
                                                dir_path=Path("./patched_images")):

                images_full_names = self.get_sorted_full_file_names(self.X_path, key = lambda x: GenerateDatasetUnet.get_num_from_name(x, 1))
                masks_full_names = self.get_sorted_full_file_names(self.Y_path, key = lambda x: GenerateDatasetUnet.get_num_from_name(x, 1))
                img_mask_full_names = zip(images_full_names, masks_full_names)

                if dir_path.exists():
                        shutil.rmtree(dir_path.name)

                os.mkdir(dir_path.name)
                curr_dir = os.getcwd()
                os.chdir(dir_path.name)
                os.mkdir("image_patches")
                os.mkdir("mask_patches")

                for img_num, img_mask_full_name in enumerate(img_mask_full_names, 1):

                        img_full_name, mask_full_name = img_mask_full_name

                        img = cv2.imread(img_full_name)
                        mask = cv2.imread(mask_full_name, as_gray=True)

                        img_patches = self.create_patches_for_image(img, crop_shape, patch_shape, is_crop=False)
                        mask_patches = self.create_patches_for_image(mask, crop_shape, patch_shape, is_crop=False)

                        for patch_num, img_mask_patch in enumerate(zip(img_patches, mask_patches), 1):

                                img_patch, mask_patch = img_mask_patch
                                img_num = str(img_num) if img_num > 9 else '0' + str(img_num)
                                patch_num = str(patch_num) if patch_num > 9 else '0' + str(patch_num)
                                com_name = img_num + "." + patch_num
                                imsave("image_patches/" + com_name + ".png", img_patch)
                                imsave("mask_patches/" + com_name + ".png", mask_patch)

                os.chdir(curr_dir)