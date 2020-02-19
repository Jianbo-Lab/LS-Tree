from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from run_classifier import ColaProcessor,MnliProcessor,MrpcProcessor,XnliProcessor,SstProcessor,ImdbProcessor, convert_examples_to_features, input_fn_builder, model_fn_builder, YelpProcessor
import numpy as np 
from time import time
# tf.logging.set_verbosity(tf.logging.INFO)

processors = {
  "cola": ColaProcessor,
  "mnli": MnliProcessor,
  "mrpc": MrpcProcessor,
  "xnli": XnliProcessor,
  "sst-2": SstProcessor,
  "imdb": ImdbProcessor,
  "yelp": YelpProcessor,
}

def construct_bert_predictor(init_checkpoint = '/home/ubuntu/bert/models/imdb_350_16_output/model.ckpt-7812', dataset_name = 'sst'):
  model_dir = './bert/models/uncased_L-12_H-768_A-12'
  bert_config_file = os.path.join(model_dir, 'bert_config.json')
  vocab_file = os.path.join(model_dir, 'vocab.txt')
  
  
  output_dir = '.'
  save_checkpoints_steps = 1000
  iterations_per_loop = 1000
  num_tpu_cores = 8
  if dataset_name == 'sst':
    if init_checkpoint is None:
      init_checkpoint = './bert/models/sst_output/model.ckpt-6313'
    max_seq_length = 128
    task_name = 'sst-2'
    batch_size = 32 

  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
  vocab_file=vocab_file, do_lower_case=True)

  tpu_cluster_resolver = None

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
  cluster=tpu_cluster_resolver,
  master=None,
  model_dir=output_dir,
  save_checkpoints_steps=save_checkpoints_steps,
  tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=iterations_per_loop,
      num_shards=num_tpu_cores,
      per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  model_fn = model_fn_builder(
  bert_config=bert_config,
  num_labels=len(label_list),
  init_checkpoint=init_checkpoint,
  learning_rate=2e-5,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=False,
  use_one_hot_embeddings=False)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
  use_tpu=False,
  model_fn=model_fn,
  config=run_config,
  train_batch_size=16,
  eval_batch_size=8,
  predict_batch_size=batch_size)

  return processor, estimator, tokenizer

def predict(xs, max_seq_length, processor, estimator, tokenizer, dataset_name): 
  if dataset_name == 'yelp':
    lines = [["0", x] for x in xs]
  else:
    lines = [[]]
    lines += [["0", x] for x in xs] 
    

  predict_examples = processor._create_examples(lines, 'test')   
  tf.logging.info("***** Running prediction*****")
  tf.logging.info("  Num examples = %d", len(predict_examples))

  features = convert_examples_to_features(predict_examples, processor.get_labels(), max_seq_length, tokenizer)

  predict_input_fn = input_fn_builder(features, seq_length = max_seq_length, is_training = False, drop_remainder = False)


  result = estimator.predict(input_fn=predict_input_fn)


  return np.array(list(result))
