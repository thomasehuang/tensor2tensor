"""Cityscapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.models import image_transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

CITYSCAPES_IMAGE_SIZE = 256
TRAIN_DIR = 'train_img'
VAL_DIR = 'val_img'

@registry.register_hparams
def imagetransformer_cityscape_uncond():
    hp = image_transformer.imagetransformerpp_base_5l_8h_big_uncond_dr00_dan_g_bs1()
    # hp.bottom["targets"] = modalities.image_channel_embeddings_bottom
    return hp

def image_generator(data_dir, training, size=CITYSCAPES_IMAGE_SIZE):
    train_prefix = TRAIN_DIR
    eval_prefix = VAL_DIR
    prefix = train_prefix if training else eval_prefix
    images_filepath = os.path.join(data_dir, prefix)
    image_files = tf.gfile.Glob(images_filepath + "/*")
    height = size
    width = size
    const_label = 0
    for filename in image_files:
        with tf.gfile.Open(filename, "rb") as f:
            encoded_image = f.read()
            yield {
                "image/encoded": [encoded_image],
                "image/format": ["png"],
                "image/class/label": [const_label],
                "image/height": [height],
                "image/width": [width]
            }

@registry.register_problem
class ImageCityscapes(image_utils.ImageProblem):

    @property
    def num_channels(self):
        return 3

    @property
    def is_small(self):
        return False

    @property
    def num_classes(self):
        return 30

    @property
    def train_shards(self):
        return 30

    @property
    def dev_shards(self):
        return 5

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        generator_utils.generate_dataset_and_shuffle(
            self.generator(data_dir, tmp_dir, True),
            self.training_filepaths(data_dir, self.train_shards, shuffled=True),
            self.generator(data_dir, tmp_dir, False),
            self.dev_filepaths(data_dir, self.dev_shards, shuffled=True))

    def generator(self, data_dir, tmp_dir, is_training):
        if is_training:
            return image_generator(
                tmp_dir, int(True), size=CITYSCAPES_IMAGE_SIZE)
        else:
            return image_generator(
                tmp_dir, int(False), size=CITYSCAPES_IMAGE_SIZE)

    def preprocess_example(self, example, mode, unused_hparams):
        example["inputs"].set_shape([CITYSCAPES_IMAGE_SIZE,
                                     CITYSCAPES_IMAGE_SIZE, 3])
        # example["inputs"] = tf.to_int64(example["inputs"])
        example["inputs"] = tf.to_float(example["inputs"])
        return example
