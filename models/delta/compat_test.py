# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for compat.py."""

import delta.compat as tf
from tensorflow.python.framework import function  # pylint:disable=g-direct-tensorflow-import


class CompatTest(tf.test.TestCase):

  def testSomeTFSymbols(self):
    self.assertFalse(tf.executing_eagerly())
    self.assertIsNotNone(tf.logging)
    self.assertIsNotNone(tf.flags)
    self.assertIs(tf.Defun, function.Defun)


if __name__ == '__main__':
  tf.test.main()
