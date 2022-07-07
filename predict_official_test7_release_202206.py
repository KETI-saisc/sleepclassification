import argparse
import glob
import importlib
import os
import sys
import numpy as np
import shutil
import sklearn.metrics as skmetrics
from sklearn.preprocessing import StandardScaler ### for standardrization
import tensorflow as tf
import re
import timeit
import tensorflow.contrib.metrics as contrib_metrics
import tensorflow.contrib.slim as contrib_slim
import nn
import math
import logging

# try:
#     os.chdir(sys._MEIPASS)
#     print(sys._MEIPASS)
# except:
#     os.chdir(os.getcwd())

logger = logging.getLogger("default_log")
_log_level = {
    None: logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

"""AASM Sleep Manual Label"""

# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1
N2 = 2      # Stage N2
N3 = 3      # Stage N3
REM = 4     # Stage REM
MOVE = 5    # Movement
UNK = 6     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}

class_dict = {
    W: "W",
    N1: "N1",
    N2: "N2",
    N3: "N3",
    REM: "REM",
    MOVE: "MOVE",
    UNK: "UNK",
}


def get_logger(
    log_file_path=None,
    name="default_log",
    level=None
):
    directory = os.path.dirname(log_file_path)
    if os.path.isdir(directory) and not os.path.exists(directory):
        os.makedirs(directory)

    root_logger = logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.baseFilename
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if any([type(h) == logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = "\x1b[36m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = logging.Formatter(f"{level_format} %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger

def print_n_samples_each_class(labels):
    """Print the number of samples in each class."""

    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        logger.info("{}: {}".format(class_dict[c], n_samples))

def load_seq_ids(fname):
    """Load sequence of IDs from txt file."""
    ids = []
    with open(fname, "r") as f:
        for line in f:
            ids.append(int(line.strip()))
    ids = np.asarray(ids)
    return ids


def iterate_batch_multiple_seq_minibatches(inputs, targets, batch_size, seq_length, shuffle_idx=None, augment_seq=False):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function randomly selects batches of multiple sequence. It then iterates
    through multiple sequence in parallel to generate a sequence of inputs and
    targets. It will append the input sequence with 0 and target with -1 when
    the lenght of each sequence is not equal.
    """

    assert len(inputs) == len(targets)
    n_inputs = len(inputs)

    if shuffle_idx is None:
        # No shuffle
        seq_idx = np.arange(n_inputs)
    else:
        # Shuffle subjects (get the shuffled indices from argument)
        seq_idx = shuffle_idx

    input_sample_shape = inputs[0].shape[1:]
    target_sample_shape = targets[0].shape[1:]

    # Compute the number of maximum loops
    n_loops = int(math.ceil(len(seq_idx) / batch_size))

    # For each batch of subjects (size=batch_size)
    for l in range(n_loops):
        start_idx = l*batch_size
        end_idx = (l+1)*batch_size
        seq_inputs = np.asarray(inputs)[seq_idx[start_idx:end_idx]]
        seq_targets = np.asarray(targets)[seq_idx[start_idx:end_idx]]

        if augment_seq:
            # Data augmentation: multiple sequences
            # Randomly skip some epochs at the beginning -> generate multiple sequence
            max_skips = 5
            for s_idx in range(len(seq_inputs)):
                n_skips = np.random.randint(max_skips)
                seq_inputs[s_idx] = seq_inputs[s_idx][n_skips:]
                seq_targets[s_idx] = seq_targets[s_idx][n_skips:]

        # Determine the maximum number of batch sequences
        n_max_seq_inputs = -1
        for s_idx, s in enumerate(seq_inputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)

        n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))

        # For each batch sequence (size=seq_length)
        for b in range(n_batch_seqs):
            start_loop = True if b == 0 else False
            start_idx = b*seq_length
            end_idx = (b+1)*seq_length
            batch_inputs = np.zeros((batch_size, seq_length) + input_sample_shape, dtype=np.float32) ## tuple 더하게 되면 행렬이 추가됨,,인수끼리 더해지는게 아님
            batch_targets = np.zeros((batch_size, seq_length) + target_sample_shape, dtype=np.int)
            batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
            batch_seq_len = np.zeros(batch_size, dtype=np.int)
            # For each subject
            for s_idx, s in enumerate(zip(seq_inputs, seq_targets)):
                # (seq_len, sample_shape)
                each_seq_inputs = s[0][start_idx:end_idx]
                each_seq_targets = s[1][start_idx:end_idx]
                batch_inputs[s_idx, :len(each_seq_inputs)] = each_seq_inputs
                batch_targets[s_idx, :len(each_seq_targets)] = each_seq_targets
                batch_weights[s_idx, :len(each_seq_inputs)] = 1
                batch_seq_len[s_idx] = len(each_seq_inputs)
            batch_x = batch_inputs.reshape((-1,) + input_sample_shape)
            batch_y = batch_targets.reshape((-1,) + target_sample_shape)
            batch_weights = batch_weights.reshape(-1)
            yield batch_x, batch_y, batch_weights, batch_seq_len, start_loop


class FHVMSleepNet(object):

    def __init__(
        self,
        config,
        output_dir="./output",
        use_rnn=False, ## true
        testing=False, ## False
        use_best=False, ## False
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rnn = use_rnn
        print("!use_rnn :", use_rnn)## true

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')  ## original
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

            if self.use_rnn: ## True
                self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None, ), name='loss_weights')##
                self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None, ), name='seq_lengths')##

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        net = self.build_cnn() ## CNN.. Ref : build_cnn()

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config: ### "n_rnn_layers" : 1
                raise Exception("Invalid config.")
            # Append the RNN if needed
            net = self.append_rnn(net)## RNN.. Ref : append_RNN()

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )

        with tf.name_scope("loss_ce_mean") as scope:
            if self.use_rnn:
                # Weight by sequence
                loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)

                # Weight by class
                sample_weights = tf.reduce_sum(
                    tf.multiply(
                        tf.one_hot(indices=self.labels, depth=self.config["n_classes"]),
                        np.asarray(self.config["class_weights"], dtype=np.float32)
                    ), 1
                )
                loss_w_class = tf.multiply(loss_w_seq, sample_weights)

                # Computer average loss scaled with the sequence length
                self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)
            else:
                self.loss_ce = tf.reduce_mean(self.loss_per_sample)

        # Regularization loss
        self.reg_losses = self.regularization_loss()

        # Total loss
        self.loss = self.loss_ce + self.reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/init_state": self.init_state,
                "train/final_state": self.final_state,

            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/init_state": self.init_state,
                "test/final_state": self.final_state,
            })

        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # 탄력적으로 gpu 메모리 사용
        self.sess = tf.Session(config=config) # sess 기반으로 학습 셋팅
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            # self.lr = tf.train.exponential_decay(
            #     learning_rate=self.config["learning_rate_decay"],
            #     global_step=self.global_step,
            #     decay_steps=self.config["decay_steps"],
            #     decay_rate=self.config["decay_rate"],
            #     staircase=False,
            #     name="learning_rate"
            # )
            self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint) ##restore model
                    #logger.info("Best model restored from {}".format(latest_checkpoint))##original
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint) ##restore model
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def build_cnn(self):
        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        with tf.variable_scope("cnn") as scope:
            net = nn.conv1d("conv1d_1", self.signals, 128, first_filter_size, first_filter_stride)
            net = nn.batch_norm("bn_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_1")

            net = nn.max_pool1d("maxpool1d_1", net, 8, 8)

            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_1")

            net = nn.conv1d("conv1d_2_1", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_1")
            net = nn.conv1d("conv1d_2_2", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_2", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_2")
            net = nn.conv1d("conv1d_2_3", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_3", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_3")

            net = nn.max_pool1d("maxpool1d_2", net, 4, 4)

            net = tf.layers.flatten(net, name="flatten_2")

        net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_2")

        return net

    def append_rnn(self, inputs):
        # RNN
        with tf.variable_scope("rnn") as scope:
            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda:tf.constant(0.5), lambda:tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # LSTM
            cells = []
            for l in range(self.config["n_rnn_layers"]):
                cells.append(_create_rnn_cell(self.config["n_rnn_units"]))

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32) ###"batch_size": 15

            # Create rnn
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )

            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]], name="reshape_nonseq_input")

            # net = tf.layers.dropout(net, rate=0.75, training=self.is_training, name="drop")

        return net

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []

        if not self.use_rnn: ## False
            for x, y in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: True,
                }

                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

                preds.extend(outputs["train/preds"])
                trues.extend(y)
        else: ## True
            for x, y, w, sl, re in minibatches:   ### train x , target y, batch_weight, batch_length, start_loop
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: True,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }

                if re:
                    # Initialize state of RNN
                    state = self.run(self.init_state)

                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

                # Buffer the final states
                state = outputs["train/final_state"]

                tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        return outputs

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []
        soft_test = []

        if not self.use_rnn: ## False
            for x, y in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: False,
                }

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)

                losses.append(outputs["test/loss"])
                preds.extend(outputs["test/preds"])
                trues.extend(y)

        else: ## True
            for x, y, w, sl, re in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: False,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }

                if re:
                    # Initialize state of RNN
                    state = self.run(self.init_state)

                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)

                soft_test.extend(self.run(tf.nn.softmax(self.logits), feed_dict=feed_dict))  ##

                # Buffer the final states
                state = outputs["test/final_state"]

                losses.append(outputs["test/loss"])

                tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
            "test/softmax": soft_test,
        }
        return outputs

    def get_current_epoch(self):
        return self.run(self.global_epoch)

    def pass_one_epoch(self):
        self.run(tf.assign(self.global_epoch, self.global_epoch+1))

    def run(self, *args, **kwargs):
        return self.sess.run(*args, **kwargs)

    def save_checkpoint(self, name):
        path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )
        logger.info("Saved checkpoint to {}".format(path))

    def save_best_checkpoint(self, name):
        path = self.best_saver.save(
            self.sess,
            os.path.join(self.best_ckpt_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )
        logger.info("Saved best checkpoint to {}".format(path))

    def save_weights(self, scope, name, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Save weights
        path = os.path.join(self.weights_path, "{}.npz".format(name))
        logger.info("Saving weights in scope: {} to {}".format(scope, path))
        save_dict = {}
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        for v in cnn_vars:
            save_dict[v.name] = self.sess.run(v)
            logger.info("  variable: {}".format(v.name))
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        np.savez(path, **save_dict)

    def load_weights(self, scope, weight_file, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Load weights
        logger.info("Loading weights in scope: {} from {}".format(scope, weight_file))
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        with np.load(weight_file) as f:
            for v in cnn_vars:
                tensor = tf.get_default_graph().get_tensor_by_name(v.name)
                self.run(tf.assign(tensor, f[v.name]))
                logger.info("  variable: {}".format(v.name))

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "cnn/conv1d_1/conv2d/kernel:0",
            "cnn/conv1d_2_1/conv2d/kernel:0",
            "cnn/conv1d_2_2/conv2d/kernel:0",
            "cnn/conv1d_2_3/conv2d/kernel:0",
            # "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0",
            # "softmax_linear/dense/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses


def load_data_bcg(subject_files, input_size):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:

            x1 = f['bcg_1']  ## Channel 1
            x2 = f['bcg_2']  ## Channel 2
            x3 = f['bcg_3']  ## Channel 3
            x4 = f['bcg_4']  ## Channel 4
            x5 = f['bcg_5']  ## Channel 5

            x_1 = x1 + x2 + x3 + x4 + x5

            ## standarization
            s = StandardScaler()
            s.fit(x_1)
            x_1 = s.transform(x_1)
            y = f['sleep_stage']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x_1 = np.squeeze(x_1)  ## Channel 1
            x_1 = x_1[:, :, np.newaxis, np.newaxis]  ## Channel 1

            temp_x = np.ones((len(x_1), input_size, 1, 1))
            temp_x[:, :, 0, 0] = x_1[:, :, 0, 0]  ## add Channel 1
            x = temp_x
            #print("Debug : X shape:", x.shape)##

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate

### To test all of Data in the data-folder when predict
def get_subject_files_sleepmat(dataset, files, sid): ### for predict
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"

    elif "sleephmc" in dataset:
        reg_exp = f"SN[1|0][0-9][0-9]_reshape+\.npz"  ##

    elif "sleepmat" in dataset:
        # reg_exp = f"[1|0][_]{str(sid)}+\.npz"
        reg_exp = f"Piezo_9Axis_2[1|2][1|0][0-9][0-9][0-9]+\.npz"  ##to drag all of sleepmat data

    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
            ##print("Debug : subject_files list")  ##print list of subject files
            ##print(subject_files)  ##

    return subject_files


def compute_performance(cm):

    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) ### sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) ### sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1


def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
    official_test_path="./data/sleepmat/official_test",##### need to be modified
):
    ##print(official_test_path,"~~~~~")##### need to be checked

    spec = importlib.util.spec_from_file_location("*", config_file)### to drag all of module in config file
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create logger
    logger = get_logger(log_file, level="info")

    subject_files = glob.glob(os.path.join(official_test_path, "*.npz")) ##

    # Load subject IDs
    fname = "{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    logger.info("Load generated SIDs from {}".format(fname))##
    logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))##

    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config["n_folds"])

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    trues = []
    preds = []
    preds_soft = []  ## modified
    F1_Score = -1
    F1_Score_sleep = -1

    for fold_idx in range(config["n_folds"]):

        #logger.info("------ Fold {}/{} ------".format(fold_idx+1, config["n_folds"]))
        logger.info("------ Processing...{}/{} -------".format(fold_idx + 1, config["n_folds"]))

        #test_sids = fold_pids[fold_idx]
        #logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

        model = FHVMSleepNet(
            config=config,
            output_dir=os.path.join(model_dir, str(fold_idx)),
            use_rnn=True,
            testing=True,
            use_best=use_best,
        )

        # Get corresponding files
        s_trues = []
        s_preds = []

        ##for sid in test_sids:
        ##logger.info("Subject ID: {}".format(fold_idx))

        #test_files = get_subject_files(
        test_files = get_subject_files_sleepmat(
            dataset=config["dataset"],
            files=subject_files,
            sid=fold_idx, # no_use
        )

        for vf in test_files : logger.info("Load files {}/{} ----------------".format(fold_idx + 1, config["n_folds"]))##
        test_x, test_y, _ = load_data_bcg(test_files, config["input_size"])

        ### Print test set
        ##logger.info("Test set (n_night_sleeps={})".format(len(test_y)))##
        ##for _x in test_x: logger.info(_x.shape)##
        ##print_n_samples_each_class(np.hstack(test_y))##

        for night_idx, night_data in enumerate(zip(test_x, test_y)):
            ### Create minibatches for testing
            night_x, night_y = night_data
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x],
                [night_y],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            if (config.get('augment_signal') is not None) and config['augment_signal']: ### False
                # Evaluate
                test_outs = model.evaluate_aug(test_minibatch_fn)
            else: ### True
                # Evaluate
                test_outs = model.evaluate(test_minibatch_fn)

            s_trues.extend(test_outs["test/trues"])
            s_preds.extend(test_outs["test/preds"])
            trues.extend(test_outs["test/trues"])
            preds.extend(test_outs["test/preds"])
            preds_soft.extend(test_outs["test/softmax"])

            # Save labels and predictions (each night of each subject)
            save_dict = {
                "y_true": test_outs["test/trues"],
                "y_pred": test_outs["test/preds"],
                "total_soft": preds_soft,##
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "pred_{}.npz".format(fname)
            )
            #np.savez(save_path, **save_dict)##
            #logger.info("Saved outputs to {}".format(save_path))##

        ##print("s_trues :", s_trues)
        ##print("s_preds :", s_preds)
        s_acc = skmetrics.accuracy_score(y_true=s_trues, y_pred=s_preds)
        s_f1_score = skmetrics.f1_score(y_true=s_trues, y_pred=s_preds, average="macro")

        s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels=[0,1,2,3,4])

        s_cm_mod = np.zeros([2,2])
        s_cm_mod[0][0] = s_cm[0][0]
        s_cm_mod[0][1] = s_cm[0][1] + s_cm[0][2]
        s_cm_mod[1][0] = s_cm[1][0] + s_cm[2][0]
        s_cm_mod[1][1] = s_cm[1][1] + s_cm[1][2] + s_cm[2][1] + s_cm[2][2]

        Recall_wake = s_cm_mod[0][0]/(s_cm_mod[0][0]+s_cm_mod[1][0])
        Precision_wake = s_cm_mod[0][0] / (s_cm_mod[0][0] + s_cm_mod[0][1])
        F1_wake = (2*Recall_wake*Precision_wake) / (Recall_wake+Precision_wake)
        Recall_sleep = s_cm_mod[1][1] / (s_cm_mod[1][1] + s_cm_mod[0][1])
        Precision_sleep = s_cm_mod[1][1] / (s_cm_mod[1][1] + s_cm_mod[1][0])
        F1_sleep = (2*Recall_sleep*Precision_sleep) / (Recall_sleep + Precision_sleep)
        s_w_f1_score = (F1_wake + F1_sleep)/2

########################################save_b_NPZ#####################################################################
        if( F1_Score_sleep < s_w_f1_score) :
            F1_Score_sleep = s_w_f1_score
            # Save labels and predictions (each night of each subject)
            save_dict = {
                "y_true": test_outs["test/trues"],
                "y_pred": test_outs["test/preds"],
                "total_soft": preds_soft,  #
                "F1_Score" : s_f1_score*100,
                "Accuracy" : s_acc*100,
                "Confusion_Matrix(row:g_true, Col:SM)": s_cm,

                "F1_Score(w/s)": s_w_f1_score * 100,
                "Confusion_Matrix_(w/s)(row:g_true, Col:SM)" : s_cm_mod,
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "predb_{}.npz".format(fname)
            )
            np.savez(save_path, **save_dict)
            if(s_w_f1_score>0.8) :
                logger.info("*** more than 80% of f1-score")

########################################save_b_NPZ######################################################################
        # if( F1_Score < s_f1_score) :
        #     F1_Score = s_f1_score
        #     ### Save labels and predictions (each night of each subject)
        #     save_dict = {
        #         "y_true": test_outs["test/trues"],
        #         "y_pred": test_outs["test/preds"],
        #         "total_soft": preds_soft,  ###
        #         "F1_Score" : s_f1_score*100,
        #         "Accuracy" : s_acc*100,
        #         "Confusion_Matrix(row:g_true, Col:SM)" : s_cm,
        #     }
        #     fname = os.path.basename(test_files[night_idx]).split(".")[0]
        #     save_path = os.path.join(
        #         output_dir,
        #         "predb_{}.npz".format(fname)
        #     )
        #     np.savez(save_path, **save_dict)
        #     if(s_f1_score>0.7) :
        #         logger.info("*****************************")

########################################################################################################################
        # logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
        #     len(s_preds),
        #     s_acc*100.0,
        #     s_f1_score*100.0,
        #     ##preds_soft, ### 시간 오래걸려 주석처리
        # ))
        #
        # logger.info(">> Confusion Matrix")
        # logger.info(s_cm)
        # ##############################################################################################################
        # save_path_total_cm = os.path.join(
        #     output_dir,
        #     "Confusion_total_{}.npz".format(fold_idx)
        # )
        # #np.savez(save_path_total_cm, s_cm)###########################################################################

########################################################################################################################

        tf.reset_default_graph()
        s_trues = []  ### initialize
        s_preds = []  ### initialize
        preds_soft = []  ### initialize

########################################################################################################################

        save_dict_total = {
            "y_true": trues,
            "y_pred": preds,
        }
        save_path_total = os.path.join(
            output_dir,
            "pred_total.npz"
        )
        #np.savez(save_path_total, **save_dict_total)##

        tf.reset_default_graph()

        logger.info("----------------------------------")
        logger.info("")

    ####################################################################################################################

    ####################################################################################################################
    # acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    # f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    # cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
    #
    # logger.info("")
    # logger.info("=== Overall ===")
    # print_n_samples_each_class(trues)
    # logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
    #     len(preds),
    #     acc*100.0,
    #     f1_score*100.0,
    # ))
    #
    # logger.info(">> Confusion Matrix")
    # logger.info(cm)
    #
    # metrics = compute_performance(cm=cm)
    # logger.info("Total: {}".format(metrics[0]))
    # logger.info("Number of samples from each class: {}".format(metrics[1]))
    # logger.info("Accuracy: {:.1f}".format(metrics[2]*100.0))
    # logger.info("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    # logger.info("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
    # logger.info("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
    # logger.info("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))
    #
    # # Save labels and predictions (all)
    # save_dict = {
    #     "y_true": trues,
    #     "y_pred": preds,
    #     "seq_sids": seq_sids,
    #     "config": config,
    # }
    # save_path = os.path.join(
    #     output_dir,
    #     "{}.npz".format(config["dataset"])
    # )
    # np.savez(save_path, **save_dict)
    # logger.info("Saved summary to {}".format(save_path))
    ####################################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=False, default="config/sleepmat.py")
    parser.add_argument("--model_dir", type=str, default="out_sleepmat1ch_onlybcg_filterdiv2_epoch300_211124/train")
    parser.add_argument("--output_dir", type=str, default="out_sleepmat1ch_onlybcg_filterdiv2_epoch300_211124/predict")
    parser.add_argument("--log_file", type=str, default="out_sleepmat1ch_onlybcg_filterdiv2_epoch300_211124/predict.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.add_argument("--official_test_path", type=str, default="./data/sleepmat/official_test") ##need to be modified..address
    parser.set_defaults(use_best=True)
    args = parser.parse_args()

    predict(
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_best=args.use_best,
        official_test_path=args.official_test_path,##
    )
