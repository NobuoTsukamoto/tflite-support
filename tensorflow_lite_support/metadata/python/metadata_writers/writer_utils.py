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
"""Helper methods for writing metadata into TFLite models."""

from typing import List, Union

from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb


def get_input_tensor_types(
    model_buffer: bytearray) -> List[_schema_fb.TensorType]:
  """Gets a list of the input tensor types."""
  subgraph = _get_subgraph(model_buffer)
  tensor_types = []
  for i in range(subgraph.InputsLength()):
    index = subgraph.Inputs(i)
    tensor_types.append(subgraph.Tensors(index).Type())
  return tensor_types


def get_output_tensor_types(
    model_buffer: bytearray) -> List[_schema_fb.TensorType]:
  """Gets a list of the output tensor types."""
  subgraph = _get_subgraph(model_buffer)
  tensor_types = []
  for i in range(subgraph.OutputsLength()):
    index = subgraph.Outputs(i)
    tensor_types.append(subgraph.Tensors(index).Type())
  return tensor_types


def load_file(file_path: str, mode: str = "b") -> Union[str, bytes]:
  """Loads file from the file path.

  Args:
    file_path: valid file path string.
    mode: a string specifies if the file should be handled as binary or text
      mode. Use "t" for text; use "b" for binary.

  Returns:
    The loaded file in str or bytes.
  """
  with open(file_path, "r" + mode) as file:
    return file.read()


def save_file(file_bytes: Union[bytes, bytearray],
              save_to_path: str,
              mode: str = "b"):
  """Loads file from the file path.

  Args:
    file_bytes: the bytes to be saved to file.
    save_to_path: valid file path string.
    mode: a string specifies if the file should be handled as binary or text
      mode. Use "t" for text; use "b" for binary.

  Returns:
    The loaded file in str or bytes.
  """
  with open(save_to_path, "w" + mode) as file:
    file.write(file_bytes)


def _get_subgraph(model_buffer: bytearray) -> _schema_fb.SubGraph:
  """Gets the subgraph of the model.

  TFLite does not support multi-subgraph. A model should have exactly one
  subgraph.

  Args:
    model_buffer: valid buffer of the model file.

  Returns:
    The subgraph of the model.

  Raises:
    ValueError: if the model has more than one subgraph or has no subgraph.
  """

  model = _schema_fb.Model.GetRootAsModel(model_buffer, 0)
  # There should be exactly one SubGraph in the model.
  if model.SubgraphsLength() != 1:
    # TODO(b/175843689): Python version cannot be specified in Kokoro bazel test
    raise ValueError("The model should have exactly one subgraph, but found " +
                     "{}.".format(model.SubgraphsLength()))

  return model.Subgraphs(0)
