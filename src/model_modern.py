"""
Modern neural network models for curiosity-driven RL.

This module provides enhanced implementations of the core neural network components
with improved type hints, documentation, and modular design.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Any, List, Optional, Tuple, Union

from constants import constants


def normalized_columns_initializer(std: float = 1.0) -> tf.keras.initializers.Initializer:
    """Create a normalized columns initializer.

    Args:
        std: Standard deviation for initialization

    Returns:
        Initializer function
    """
    def _initializer(shape: Tuple[int, ...], dtype: Optional[tf.DType] = None, partition_info: Optional[Any] = None) -> tf.Tensor:
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def cosine_loss(A: tf.Tensor, B: tf.Tensor, name: str) -> tf.Tensor:
    """Compute cosine loss between two tensors.

    Args:
        A: First tensor (BatchSize, d)
        B: Second tensor (BatchSize, d)
        name: Name for the loss operation

    Returns:
        Cosine loss tensor
    """
    dotprod = tf.reduce_sum(
        tf.multiply(tf.nn.l2_normalize(A, 1), tf.nn.l2_normalize(B, 1)), 1
    )
    loss = 1 - tf.reduce_mean(dotprod, name=name)
    return loss


def flatten(x: tf.Tensor) -> tf.Tensor:
    """Flatten tensor to 2D.

    Args:
        x: Input tensor

    Returns:
        Flattened tensor
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(
    x: tf.Tensor,
    num_filters: int,
    name: str,
    filter_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    pad: str = "SAME",
    dtype: tf.DType = tf.float32,
    collections: Optional[List[str]] = None,
) -> tf.Tensor:
    """2D convolution layer with Xavier initialization.

    Args:
        x: Input tensor
        num_filters: Number of output filters
        name: Variable scope name
        filter_size: Filter size (height, width)
        stride: Stride (height, width)
        pad: Padding type
        dtype: Data type
        collections: Variable collections

    Returns:
        Convolved tensor
    """
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # Xavier initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * num_filters
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable(
            "W",
            filter_shape,
            dtype,
            tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections
        )
        b = tf.get_variable(
            "b",
            [1, 1, 1, num_filters],
            initializer=tf.constant_initializer(0.0),
            collections=collections
        )
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def deconv2d(
    x: tf.Tensor,
    out_shape: List[int],
    name: str,
    filter_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    pad: str = "SAME",
    dtype: tf.DType = tf.float32,
    collections: Optional[List[str]] = None,
    prev_num_feat: Optional[int] = None,
) -> tf.Tensor:
    """2D transposed convolution layer.

    Args:
        x: Input tensor
        out_shape: Output shape
        name: Variable scope name
        filter_size: Filter size (height, width)
        stride: Stride (height, width)
        pad: Padding type
        dtype: Data type
        collections: Variable collections
        prev_num_feat: Previous number of features

    Returns:
        Deconvolved tensor
    """
    with tf.variable_scope(name):
        num_filters = out_shape[-1]
        prev_num_feat = int(x.get_shape()[3]) if prev_num_feat is None else prev_num_feat
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], num_filters, prev_num_feat]

        # Xavier initialization
        fan_in = np.prod(filter_shape[:2]) * prev_num_feat
        fan_out = np.prod(filter_shape[:3])
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable(
            "W",
            filter_shape,
            dtype,
            tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections
        )
        b = tf.get_variable(
            "b",
            [num_filters],
            initializer=tf.constant_initializer(0.0),
            collections=collections
        )
        deconv2d = tf.nn.conv2d_transpose(x, w, tf.pack(out_shape), stride_shape, pad)
        return deconv2d


def linear(
    x: tf.Tensor,
    size: int,
    name: str,
    initializer: Optional[tf.keras.initializers.Initializer] = None,
    bias_init: float = 0,
) -> tf.Tensor:
    """Linear layer.

    Args:
        x: Input tensor
        size: Output size
        name: Variable scope name
        initializer: Weight initializer
        bias_init: Bias initialization value

    Returns:
        Linear transformation result
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits: tf.Tensor, d: int) -> tf.Tensor:
    """Sample from categorical distribution.

    Args:
        logits: Logits tensor
        d: Number of categories

    Returns:
        One-hot sampled tensor
    """
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def inverse_universe_head(x: tf.Tensor, final_shape: List[int], n_convs: int = 4) -> tf.Tensor:
    """Inverse universe head for state prediction.

    Args:
        x: Input features [None, 288]
        final_shape: Target output shape
        n_convs: Number of convolution layers

    Returns:
        Reconstructed state [None, height, width, channels]
    """
    print('Using inverse-universe head design')
    bs = tf.shape(x)[0]
    deconv_shape1 = [final_shape[1]]
    deconv_shape2 = [final_shape[2]]

    for i in range(n_convs):
        deconv_shape1.append((deconv_shape1[-1] - 1) // 2 + 1)
        deconv_shape2.append((deconv_shape2[-1] - 1) // 2 + 1)

    inshapeprod = np.prod(x.get_shape().as_list()[1:]) / 32.0
    assert inshapeprod == deconv_shape1[-1] * deconv_shape2[-1]

    x = tf.reshape(x, [-1, deconv_shape1[-1], deconv_shape2[-1], 32])
    deconv_shape1 = deconv_shape1[:-1]
    deconv_shape2 = deconv_shape2[:-1]

    for i in range(n_convs - 1):
        x = tf.nn.elu(deconv2d(
            x,
            [bs, deconv_shape1[-1], deconv_shape2[-1], 32],
            "dl{i + 1}",
            [3, 3],
            [2, 2],
            prev_num_feat=32
        ))
        deconv_shape1 = deconv_shape1[:-1]
        deconv_shape2 = deconv_shape2[:-1]

    x = deconv2d(x, [bs] + final_shape[1:], "dl4", [3, 3], [2, 2], prev_num_feat=32)
    return x


def universe_head(x: tf.Tensor, n_convs: int = 4) -> tf.Tensor:
    """Universe head for feature extraction.

    Args:
        x: Input state [None, height, width, channels]
        n_convs: Number of convolution layers

    Returns:
        Features [None, 288]
    """
    print('Using universe head design')
    for i in range(n_convs):
        x = tf.nn.elu(conv2d(x, 32, "l{i + 1}", [3, 3], [2, 2]))
    x = flatten(x)
    return x


def nips_head(x: tf.Tensor) -> tf.Tensor:
    """NIPS 2013 DQN head.

    Args:
        x: Input state [None, 84, 84, 4]

    Returns:
        Features [None, 256]
    """
    print('Using nips head design')
    x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x


def nature_head(x: tf.Tensor) -> tf.Tensor:
    """Nature 2015 DQN head.

    Args:
        x: Input state [None, 84, 84, 4]

    Returns:
        Features [None, 512]
    """
    print('Using nature head design')
    x = tf.nn.relu(conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 512, "fc", normalized_columns_initializer(0.01)))
    return x


def doom_head(x: tf.Tensor) -> tf.Tensor:
    """Doom-specific head.

    Args:
        x: Input state [None, 120, 160, 1]

    Returns:
        Features [None, 256]
    """
    print('Using doom head design')
    x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
    x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
    x = flatten(x)
    x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x


class LSTMPolicy:
    """LSTM-based policy network for A3C."""

    def __init__(
        self,
        ob_space: Tuple[int, ...],
        ac_space: int,
        design_head: str = 'universe'
    ):
        """Initialize LSTM policy.

        Args:
            ob_space: Observation space shape
            ac_space: Action space size
            design_head: Network head design
        """
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        size = 256

        # Apply head design
        if design_head == 'nips':
            x = nips_head(x)
        elif design_head == 'nature':
            x = nature_head(x)
        elif design_head == 'doom':
            x = doom_head(x)
        elif 'tile' in design_head:
            x = universe_head(x, n_convs=2)
        else:
            x = universe_head(x)

        # LSTM layer
        x = tf.expand_dims(x, [0])  # Add batch dimension
        lstm = tf.keras.layers.LSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        # Initialize LSTM state
        c_init = np.zeros((1, lstm.state_size[0]), np.float32)
        h_init = np.zeros((1, lstm.state_size[1]), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size[0]], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size[1]], name='h_in')
        self.state_in = [c_in, h_in]

        # Run LSTM
        state_in = [c_in, h_in]
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False
        )
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # Output layers
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # Policy outputs
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]

        # Variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self) -> List[np.ndarray]:
        """Get initial LSTM features.

        Returns:
            Initial LSTM state
        """
        return self.state_init

    def act(
        self,
        ob: np.ndarray,
        c: np.ndarray,
        h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get action from policy.

        Args:
            ob: Current observation
            c: LSTM cell state
            h: LSTM hidden state

        Returns:
            Tuple of (action, value, new_c, new_h)
        """
        sess = tf.get_default_session()
        return sess.run(
            [self.sample, self.vf] + self.state_out,
            {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h}
        )

    def act_inference(
        self,
        ob: np.ndarray,
        c: np.ndarray,
        h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get action with probabilities for inference.

        Args:
            ob: Current observation
            c: LSTM cell state
            h: LSTM hidden state

        Returns:
            Tuple of (probs, sample, value, new_c, new_h)
        """
        sess = tf.get_default_session()
        return sess.run(
            [self.probs, self.sample, self.vf] + self.state_out,
            {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h}
        )

    def value(self, ob: np.ndarray, c: np.ndarray, h: np.ndarray) -> float:
        """Get value estimate.

        Args:
            ob: Current observation
            c: LSTM cell state
            h: LSTM hidden state

        Returns:
            Value estimate
        """
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class StateActionPredictor:
    """State-action predictor for ICM."""

    def __init__(
        self,
        ob_space: Tuple[int, ...],
        ac_space: int,
        design_head: str = 'universe'
    ):
        """Initialize state-action predictor.

        Args:
            ob_space: Observation space shape
            ac_space: Action space size
            design_head: Network head design
        """
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # Feature encoding
        size = 256
        if design_head == 'nips':
            phi1 = nips_head(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = nips_head(phi2)
        elif design_head == 'nature':
            phi1 = nature_head(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = nature_head(phi2)
        elif design_head == 'doom':
            phi1 = doom_head(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = doom_head(phi2)
        elif 'tile' in design_head:
            phi1 = universe_head(phi1, n_convs=2)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universe_head(phi2, n_convs=2)
        else:
            phi1 = universe_head(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universe_head(phi2)

        # Inverse model: g(phi1,phi2) -> a_inv
        g = tf.concat(1, [phi1, phi2])
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, aindex),
            name="invloss"
        )
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)

        # Forward model: f(phi1,asample) -> phi2
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        self.forwardloss = 0.5 * tf.reduce_mean(
            tf.square(tf.subtract(f, phi2)),
            name='forwardloss'
        )
        self.forwardloss = self.forwardloss * 288.0  # Scale by feature dimension

        # Variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Predict action from states.

        Args:
            s1: First state
            s2: Second state

        Returns:
            Predicted action probabilities
        """
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1: np.ndarray, s2: np.ndarray, asample: np.ndarray) -> float:
        """Predict curiosity bonus.

        Args:
            s1: First state
            s2: Second state
            asample: Action taken

        Returns:
            Curiosity bonus
        """
        sess = tf.get_default_session()
        error = sess.run(
            self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]}
        )
        error = error * constants['PREDICTION_BETA']
        return error


class StatePredictor:
    """State predictor for ICM baseline."""

    def __init__(
        self,
        ob_space: Tuple[int, ...],
        ac_space: int,
        design_head: str = 'universe',
        unsup_type: str = 'state'
    ):
        """Initialize state predictor.

        Args:
            ob_space: Observation space shape
            ac_space: Action space size
            design_head: Network head design
            unsup_type: Type of unsupervised learning
        """
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])
        self.state_aenc = unsup_type == 'stateAenc'

        # Feature encoding
        if design_head == 'universe':
            phi1 = universe_head(phi1)
            if self.state_aenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universe_head(phi2)
        elif 'tile' in design_head:
            phi1 = universe_head(phi1, n_convs=2)
            if self.state_aenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universe_head(phi2)
        else:
            raise ValueError("Only universe designHead implemented for state prediction baseline.")

        # Forward model: f(phi1,asample) -> phi2
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, phi1.get_shape()[1].value, "f1", normalized_columns_initializer(0.01)))
        if 'tile' in design_head:
            f = inverse_universe_head(f, input_shape, n_convs=2)
        else:
            f = inverse_universe_head(f, input_shape)

        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        if self.state_aenc:
            self.aenc_bonus = 0.5 * tf.reduce_mean(
                tf.square(tf.subtract(phi1, phi2_aenc)),
                name='aencBonus'
            )
        self.predstate = phi1

        # Variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_state(self, s1: np.ndarray, asample: np.ndarray) -> np.ndarray:
        """Predict next state.

        Args:
            s1: Current state
            asample: Action taken

        Returns:
            Predicted next state
        """
        sess = tf.get_default_session()
        return sess.run(
            self.predstate,
            {self.s1: [s1], self.asample: [asample]}
        )[0, :]

    def pred_bonus(self, s1: np.ndarray, s2: np.ndarray, asample: np.ndarray) -> float:
        """Predict curiosity bonus.

        Args:
            s1: First state
            s2: Second state
            asample: Action taken

        Returns:
            Curiosity bonus
        """
        sess = tf.get_default_session()
        bonus = self.aenc_bonus if self.state_aenc else self.forwardloss
        error = sess.run(
            bonus,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]}
        )
        error = error * constants['PREDICTION_BETA']
        return error
