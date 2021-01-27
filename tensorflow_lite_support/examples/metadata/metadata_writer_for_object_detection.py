# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Writes metadata and label file to the image classifier models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Union

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.platform import resource_loader

import flatbuffers
# pylint: disable=g-direct-tensorflow-import
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.metadata_writers import object_detector
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils
# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string("model_file", None,
                      "Path and file name to the TFLite model file.")
  flags.DEFINE_string("label_file", None, "Path to the label file.")
  flags.DEFINE_string("export_directory", None,
                      "Path to save the TFLite model files with metadata.")
  flags.mark_flag_as_required("model_file")
  flags.mark_flag_as_required("label_file")
  flags.mark_flag_as_required("export_directory")


class ModelSpecificInfo(object):
  """Holds information that is specificly tied to an image classifier."""

  def __init__(self, name, version, image_width, image_height, image_min,
               image_max, mean, std):
    self.name = name
    self.version = version
    self.image_width = image_width
    self.image_height = image_height
    self.image_min = image_min
    self.image_max = image_max
    self.mean = mean
    self.std = std


_MODEL_INFO = {
    "ssdlite_mobiledet_cpu":
        ModelSpecificInfo(
            name="SSDLite with MobileDet-CPU",
            version="v1",
            image_width=320,
            image_height=320,
            image_min=0,
            image_max=255,
            mean=[127.5],
            std=[127.5])
}


class MetadataPopulatorForObjectDetector(object):
  """Populates the metadata for an object detector."""

  def __init__(self, model_file_path, export_model_path, model_info, label_file_path):
    self.model_file_path = model_file_path
    self.export_model_path = export_model_path
    self.model_info = model_info
    self.label_file_path = [label_file_path]
    self.model_buffer = None
    self.metadata_buf = None
    self.general_md = None
    self.input_md = None
    self.output_category_md = None
    self.writer = None


  def populate(self):
    """Creates metadata and then populates it for an image classifier."""
    self._create_metadata()
    self._populate_metadata()

  def get_metadata_json(self):
    return self.writer.get_metadata_json()

  def _load_file(self, file_name: str, mode: str = "rb") -> Union[str, bytes]:
    """Loads files from resources."""
    file_path = resource_loader.get_path_to_datafile(file_name)
    with open(file_path, mode) as file:
      return file.read()


  def _create_metadata(self):
    """Creates the metadata for an image classifier."""

    # Creates model info.
    self.general_md = metadata_info.GeneralMd(
      name=self.model_info.name,
      version=self.model_info.version,
      description=("Identify which of a known set of objects might be present "
                   "and provide information about their positions within the "
                   "given image or a video stream."),
      author="Test",
      licenses=("Apache License. Version 2.0 "
                 "http://www.apache.org/licenses/LICENSE-2.0.")
    )

    # Load model to buffer.
    self.model_buffer = self._load_file(self.model_file_path)

    # Creates input info.
    self.input_md = metadata_info.InputImageTensorMd(
      name="normalized_input_image_tensor",
      description=("Input image to be classified. The expected image is {0} x {1}, with "
                   "three channels (red, blue, and green) per pixel. Each value in the "
                   "tensor is a single byte between {2} and {3}.".format(
                     self.model_info.image_width, self.model_info.image_height,
                     self.model_info.image_min, self.model_info.image_max)),
        norm_mean=self.model_info.mean,
        norm_std=self.model_info.std,
        color_space_type=_metadata_fb.ColorSpaceType.RGB,
        tensor_type=writer_utils.get_input_tensor_types(self.model_buffer)[0])

    # Creates output info.
    self.output_category_md = metadata_info.CategoryTensorMd(
        name="category",
        description="The categories of the detected boxes.",
        label_files=[
            metadata_info.LabelFileMd(file_path=file_path)
            for file_path in self.label_file_path
        ])


  def _populate_metadata(self):
    """Populates metadata and label file to the model file."""
    self.writer = object_detector.MetadataWriter.create_from_metadata_info(
        model_buffer=self.model_buffer, general_md=self.general_md,
        input_md=self.input_md, output_category_md=self.output_category_md)
    model_with_metadata = self.writer.populate()

    with open(self.export_model_path, "wb") as f:
      f.write(model_with_metadata)


def main(_):
  model_file = FLAGS.model_file
  model_basename = os.path.splitext(os.path.basename(model_file))[0]
  if model_basename not in _MODEL_INFO:
    raise ValueError(
        "The model info for, {0}, is not defined yet.".format(model_basename))

  export_model_path = os.path.join(FLAGS.export_directory, model_basename)

  # Generate the metadata objects and put them in the model file
  populator = MetadataPopulatorForObjectDetector(
      FLAGS.model_file, export_model_path, _MODEL_INFO.get(model_basename), FLAGS.label_file)
  populator.populate()

  # Validate the output model file by reading the metadata and produce
  # a json file with the metadata under the export path
  export_json_file = os.path.join(FLAGS.export_directory,
                                  os.path.splitext(model_basename)[0] + ".json")
  json_file = populator.get_metadata_json()
  with open(export_json_file, "w") as f:
    f.write(json_file)

  print("Finished populating metadata and associated file to the model:")
  print(model_file)
  print("The metadata json file has been saved to:")
  print(export_json_file)


if __name__ == "__main__":
  define_flags()
  app.run(main)
