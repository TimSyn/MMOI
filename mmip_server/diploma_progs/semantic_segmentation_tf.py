from unet_model_tf import *
from pathlib import Path
import numpy as np
import json
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import datetime
from given_unet import res_unet

# activate eager mode
# tf.compat.v1.enable_eager_execution()
# print(tf.executing_eagerly())

batch_size = 6
input_shape = (384, 384)
epochs_num = 30
shuffle_size = 700
rotations_amount = 18
amount_of_classes = 12 + 1# for background + 1 class

# creating datasets
train_sample_path = Path("train_plain/img_patches")
train_mask_path = Path("train_plain/mask_patches")
train_weight_map_path = Path("train_plain/weight_maps")

valid_sample_path = Path("test_plain/img_patches")
valid_mask_path = Path("test_plain/mask_patches")
valid_weight_map_path = Path("test_plain/weight_maps")

test_sample_path = Path("/Users/macbookpro/Downloads/test_full_size_plain copy")
test_mask_path = Path("/Users/macbookpro/Downloads/test_full_size_plain copy")
test_weight_map_path = Path("../test/weight_maps")

train = GenerateDatasetUnet(amount_of_classes, train_sample_path, train_mask_path, weight_map_path=train_weight_map_path)()
valid = GenerateDatasetUnet(amount_of_classes, valid_sample_path, valid_mask_path, weight_map_path=valid_weight_map_path)(is_augment=False, identity_weight_maps=True)
# # test = GenerateDatasetUnet(amount_of_classes, test_sample_path, test_mask_path, weight_map_path=test_weight_map_path, shuffle=False)()

train_size = len(train)
train = train.batch(batch_size).repeat()
# test = test.batch(batch_size)
valid = valid.batch(batch_size)

# Creating checkpoints to save weights during training
checkpoint_dir = Path("./checkpoints")

if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir.name)

checkpoint_path = "./checkpoints/epoch_{epoch:03d}_{val_loss:.3f}.cpkt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, epsilon=1e-4)

if not Path("./logs").exists():
        os.mkdir("./logs")

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

img_input = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3 * rotations_amount), name='img_input', dtype=tf.float32)
weight_map_input = tf.keras.layers.Input(shape=input_shape, name='weight_map_input', dtype=tf.float32)

new_unet = res_unet((input_shape[0], input_shape[1], 3 * rotations_amount), input_shape, amount_of_classes, filters=64)

# unet = UnetNN(in_classes=amount_of_classes, L2_const=0.0001, drop_prob=0.0, n_filters=64)
# unet(inputs=[img_input, weight_map_input])

schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.00005, decay_rate=0.8,
                                                          decay_steps=int(train_size/batch_size))
# unet.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#         # run_eagerly=True, metrics={"output_1" : PlugMetric(), "output_2" : MeanIoUMetric()},
#         run_eagerly=True, metrics={"output_2" : [AccuracyMetric(), SeparateIoU()]},
#         loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=False), PlugLoss()],
#         loss_weights=[1.0, 0.0])

new_unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=schedule),
        # run_eagerly=True, metrics={"output_1" : PlugMetric(), "output_2" : MeanIoUMetric()},
        run_eagerly=True, metrics=[AccuracyMetric(), SeparateIoU(print_to_file=True)],
        loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=False)])

# print(new_unet.summary())
# plot_model(unet, to_file="my_model.png", show_shapes=True, show_layer_names=True)
# inference
if len(os.listdir(checkpoint_dir.name)) != 0:
        # unet.load_weights("checkpoints/epoch_028_0.918.cpkt") # non-pol 0.8055 0.68
        new_unet.load_weights("checkpoints/epoch_027_0.272.cpkt") # non-pol 0.8055 0.68
        # unet.load_weights("checkpoints/epoch_030_0.792.cpkt") # non-pol new 0.
#         # unet.load_weights("checkpoints/epoch_030_0.803.cpkt") # pol 0.81 0.723
        print("Weights loaded.")

# print(unet.trainable_variables[0])

# res = unet.test(batch_size, test_sample_path, test_mask_path)

# training
# history = unet.fit(train,
#                 batch_size=batch_size,N
#                 epochs=epochs_num, verbose=1,
#                 steps_per_epoch=train_size//batch_size,
#                 callbacks=[cp_callback, lr_callback],
#                 validation_data=valid)

history = new_unet.fit(train,
                batch_size=batch_size,
                epochs=epochs_num, verbose=1,
                steps_per_epoch=train_size//batch_size,
                callbacks=[cp_callback],
                validation_data=valid)

# evaluation
# unet.evaluate(test, verbose=1, batch_size=batch_size)

# saving
with open("loss_values.txt", 'wt') as f:
        tmp = str(history.history['loss'])[1:-1]
        f.write(tmp)

with open("val_loss_values.txt", 'wt') as f:
        tmp = str(history.history['val_loss'])[1:-1]
        f.write(tmp)