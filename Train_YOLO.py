"""
MODIFIED FROM keras-yolo3 PACKAGE, https://github.com/qqwweee/keras-yolo3
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import argparse
import warnings


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(0), "src")
sys.path.append(src_path)

utils_path = os.path.join(get_parent_dir(1), "Utils")
sys.path.append(utils_path)

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from keras_yolo3.yolo3.model import (
    preprocess_true_boxes,
    yolo_body,
    tiny_yolo_body,
    yolo_loss,
)
from keras_yolo3.yolo3.utils import get_random_data
from PIL import Image
from time import time
import tensorflow.compat.v1 as tf
import pickle

from Train_Utils import (
    get_classes,
    get_anchors,
    create_model,
    create_tiny_model,
    data_generator,
    data_generator_wrapper,
    ChangeToOtherMachine,
)


keras_path = './keras_yolo3/'
Image_Folder = './images/'
YOLO_filename = './annotations/data_train.txt'
custom_classname = './annotations/data_classes.txt'
log_dir = './log_dir/'
anchors_path = keras_path + 'model_data/yolo_anchors.txt'
weights_path = keras_path + 'yolov3.h5'

if __name__ == "__main__":
    class_names = get_classes(custom_classname)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)  # multiple of 32, height, width
    epoch1, epoch2 = 1, 1

    is_tiny_version = len(anchors) == 6  # default setting

    model = create_model(
        input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path
    )  # make sure you know what you freeze

    log_dir_time = os.path.join(log_dir, "{}".format(int(time())))
    logging = TensorBoard(log_dir=log_dir_time)
    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, "checkpoint.h5"),
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        period=5,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1
    )

    val_split = 0.1
    with open(YOLO_filename) as f:
        lines = f.readlines()

    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    frozen_callbacks = [logging, checkpoint]


    model.compile(
        optimizer=Adam(lr=1e-3),
        loss={
            # use custom yolo_loss Lambda layer.
            "yolo_loss": lambda y_true, y_pred: y_pred
        },
    )

    batch_size = 32
    print(
        "Train on {} samples, val on {} samples, with batch size {}.".format(
            num_train, num_val, batch_size
        )
    )
    history = model.fit(
        data_generator_wrapper(
            lines[:num_train], batch_size, input_shape, anchors, num_classes
        ),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator_wrapper(
            lines[num_train:], batch_size, input_shape, anchors, num_classes
        ),
        validation_steps=max(1, num_val // batch_size),
        epochs=epoch1,
        initial_epoch=0,
        callbacks=frozen_callbacks,
    )
    model.save_weights(log_dir + "trained_weights_stage_1.h5")

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is unsatisfactory.

    # full_callbacks = [logging, checkpoint, reduce_lr, early_stopping]
    #
    #
    # for i in range(len(model.layers)):
    #     model.layers[i].trainable = True
    # model.compile(
    #     optimizer=Adam(lr=1e-4), loss={"yolo_loss": lambda y_true, y_pred: y_pred}
    # )  # recompile to apply the change
    #
    # print("Unfreeze all layers.")
    #
    # batch_size = 4  # note that more GPU memory is required after unfreezing the body
    # print(
    #     "Train on {} samples, val on {} samples, with batch size {}.".format(
    #         num_train, num_val, batch_size
    #     )
    # )
    # history = model.fit(
    #     data_generator_wrapper(
    #         lines[:num_train], batch_size, input_shape, anchors, num_classes
    #     ),
    #     steps_per_epoch=max(1, num_train // batch_size),
    #     validation_data=data_generator_wrapper(
    #         lines[num_train:], batch_size, input_shape, anchors, num_classes
    #     ),
    #     validation_steps=max(1, num_val // batch_size),
    #     epochs=epoch1 + epoch2,
    #     initial_epoch=epoch1,
    #     callbacks=full_callbacks,
    # )
    # model.save_weights(log_dir + "trained_weights_final.h5")
