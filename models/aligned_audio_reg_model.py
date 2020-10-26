import numpy as np
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import delta.compat as tf
import delta.layers
from delta.models.base_model import Model

from delta.utils.register import registers
from delta.layers.utils import get_expand_pad_mask
import pickle

@registers.model.register
class AlignClassModel(Model):
  def __init__(self,dropout,units):
    super().__init__()
    self.max_text_len = 516
    self.hidden_dim = units
    self.head_num = 5
    self.dropout_rate = dropout
    self.speech_dropout_rate = dropout
    self.speech_dense_act = 'relu'


    
    self.speech_enc_layer = delta.layers.RnnEncoder(dropout = dropout,units = units, name="speech_encoder")
    self.text_enc_layer = delta.layers.RnnEncoder(dropout = dropout,units = units, name="text_encoder")

    self.align_attn_layer = delta.layers.MultiHeadAttention(
      self.hidden_dim, self.head_num)

    self.align_enc_layer = delta.layers.RnnAttentionEncoder(dropout = dropout,units = units,
       name="align_encoder")



    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.speech_d = tf.keras.layers.Dropout(self.speech_dropout_rate)
    self.speech_enc_d = tf.keras.layers.Dropout(self.speech_dropout_rate)
    self.text_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.attn_enc_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.align_enc_d = tf.keras.layers.Dropout(self.dropout_rate)

    

    self.final_dense = tf.keras.layers.Dense(
      1,
      activation=tf.keras.activations.linear)

    self.align_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.speech_dense = tf.keras.layers.Dense(
      512, activation=self.speech_dense_act)

  def build(self, input_shape):

    self.concatenate = tf.keras.layers.Concatenate()
    self.reshape2 = tf.keras.layers.Reshape(
      (self.max_text_len, self.hidden_dim),
      input_shape=(-1, self.hidden_dim))
    self.built = True

  def call(self, inputs, training=None, mask=None):

    speechs = inputs[1]
    texts = inputs[0]

    text_enc, _ = self.text_enc_layer(texts,
                                  training=training,
                                  mask=None)

    text_enc = self.text_enc_d(text_enc, training=training)

    speech_enc, _ = self.speech_enc_layer(speechs,
                                          training=training)
    speech_enc = self.speech_enc_d(speech_enc, training=training)

    attn_outs, _ = self.align_attn_layer(
      (text_enc, speech_enc, speech_enc),
      training=training)
    attn_outs = self.attn_enc_d(attn_outs, training=training)
    attn_outs = self.reshape2(attn_outs)

    align_enc = self.align_enc_layer(attn_outs,
                                     training=training,
                                     mask=None)

    align_enc_dropped = self.align_enc_d(align_enc, training=training)

    scores = self.final_dense(align_enc_dropped)
    return scores,align_enc




