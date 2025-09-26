"""
Modern A3C implementation with integrated logging and gymnasium support.

This module provides an enhanced version of the A3C algorithm with:
- Weights & Biases and TensorBoard logging
- Gymnasium environment support
- Type hints and improved documentation
- Modular design for easy extension
"""
from __future__ import annotations

import queue
import threading
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy import signal

from constants import constants
from logger import Logger
from model_modern import LSTMPolicy, StateActionPredictor, StatePredictor

# Check TensorFlow version for compatibility
use_tf12_api = tf.__version__ >= "2.0.0"


def discount(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns.

    Args:
        x: Array of rewards [r1, r2, r3, ..., rN]
        gamma: Discount factor

    Returns:
        Array of discounted returns [r1 + r2*gamma + r3*gamma^2 + ...,
                                   r2 + r3*gamma + r4*gamma^2 + ...,
                                   ...]
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(
    rollout: "PartialRollout",
    gamma: float,
    lambda_: float = 1.0,
    clip: bool = False
) -> "Batch":
    """Process a rollout to compute returns and advantages.

    Args:
        rollout: The rollout to process
        gamma: Discount factor
        lambda_: GAE lambda parameter
        clip: Whether to clip rewards

    Returns:
        Processed batch with advantages and returns
    """
    # Collect transitions
    if rollout.unsup:
        batch_si = np.asarray(rollout.states + [rollout.end_state])
    else:
        batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)

    # Collect target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(rollout.bonuses + [0])
    if clip:
        rewards_plus_v[:-1] = np.clip(
            rewards_plus_v[:-1],
            -constants['REWARD_CLIP'],
            constants['REWARD_CLIP']
        )
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # value network target

    # Collect target for policy network
    rewards = np.asarray(rollout.rewards)
    if rollout.unsup:
        rewards += np.asarray(rollout.bonuses)
    if clip:
        rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])

    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class PartialRollout:
    """A piece of a complete rollout for experience collection."""

    def __init__(self, unsup: bool = False):
        """Initialize rollout.

        Args:
            unsup: Whether this rollout uses unsupervised learning
        """
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.r = 0.0
        self.terminal = False
        self.features: List[Tuple[np.ndarray, np.ndarray]] = []
        self.unsup = unsup
        if self.unsup:
            self.bonuses: List[float] = []
            self.end_state: Optional[np.ndarray] = None

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        terminal: bool,
        features: Tuple[np.ndarray, np.ndarray],
        bonus: Optional[float] = None,
        end_state: Optional[np.ndarray] = None,
    ) -> None:
        """Add a transition to the rollout.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            terminal: Whether episode is terminal
            features: LSTM features
            bonus: Intrinsic reward bonus (if using unsupervised learning)
            end_state: Next state (if using unsupervised learning)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminal = terminal
        self.features.append(features)
        if self.unsup:
            self.bonuses.append(bonus or 0.0)
            self.end_state = end_state

    def extend(self, other: "PartialRollout") -> None:
        """Extend this rollout with another rollout.

        Args:
            other: Another rollout to extend with
        """
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if self.unsup:
            self.bonuses.extend(other.bonuses)
            self.end_state = other.end_state


class RunnerThread(threading.Thread):
    """Thread for running environment interactions."""

    def __init__(
        self,
        env: Any,
        policy: LSTMPolicy,
        num_local_steps: int,
        visualise: bool,
        predictor: Optional[Union[StateActionPredictor, StatePredictor]],
        env_wrap: bool,
        no_reward: bool,
        logger: Optional[Logger] = None,
    ):
        """Initialize runner thread.

        Args:
            env: Environment to run
            policy: Policy network
            num_local_steps: Number of local steps per rollout
            visualise: Whether to visualize
            predictor: Curiosity predictor (optional)
            env_wrap: Whether environment is wrapped
            no_reward: Whether to remove rewards
            logger: Logger for metrics
        """
        super().__init__()
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.predictor = predictor
        self.env_wrap = env_wrap
        self.no_reward = no_reward
        self.logger = logger

    def start_runner(self, sess: tf.Session, summary_writer: Any) -> None:
        """Start the runner thread.

        Args:
            sess: TensorFlow session
            summary_writer: Summary writer for TensorBoard
        """
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self) -> None:
        """Run the environment interaction loop."""
        with self.sess.as_default():
            self._run()

    def _run(self) -> None:
        """Main environment interaction loop."""
        rollout_provider = env_runner(
            self.env,
            self.policy,
            self.num_local_steps,
            self.summary_writer,
            self.visualise,
            self.predictor,
            self.env_wrap,
            self.no_reward,
            self.logger,
        )
        while True:
            # The timeout variable exists because apparently, if one worker dies,
            # the other workers won't die with it, unless the timeout is set to
            # some large number. This is an empirical observation.
            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(
    env: Any,
    policy: LSTMPolicy,
    num_local_steps: int,
    summary_writer: Any,
    render: bool,
    predictor: Optional[Union[StateActionPredictor, StatePredictor]],
    env_wrap: bool,
    no_reward: bool,
    logger: Optional[Logger] = None,
):
    """Environment runner for collecting experience.

    Args:
        env: Environment to run
        policy: Policy network
        num_local_steps: Number of local steps per rollout
        summary_writer: Summary writer for TensorBoard
        render: Whether to render
        predictor: Curiosity predictor (optional)
        env_wrap: Whether environment is wrapped
        no_reward: Whether to remove rewards
        logger: Logger for metrics

    Yields:
        PartialRollout: Collected experience rollouts
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    episode_start_time = time.time()

    if predictor is not None:
        ep_bonus = 0
        life_bonus = 0

    while True:
        terminal_end = False
        rollout = PartialRollout(predictor is not None)

        for _ in range(num_local_steps):
            # Run policy
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # Run environment: get action_index from sampled one-hot 'action'
            step_act = action.argmax()
            state, reward, terminal, truncated, info = env.step(step_act)

            if no_reward:
                reward = 0.0
            if render:
                env.render()

            curr_tuple = [last_state, action, reward, value_, terminal or truncated, last_features]
            if predictor is not None:
                bonus = predictor.pred_bonus(last_state, state, action)
                curr_tuple += [bonus, state]
                life_bonus += bonus
                ep_bonus += bonus

            # Collect the experience
            rollout.add(*curr_tuple)
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features

            timestep_limit = getattr(env.spec, 'max_episode_steps', None)
            if timestep_limit is None:
                timestep_limit = getattr(env.spec, 'timestep_limit', None)
            if terminal or truncated or (timestep_limit and length >= timestep_limit):
                # Log episode summary
                episode_time = time.time() - episode_start_time
                if logger:
                    logger.log_episode_summary(
                        step=policy.global_step.eval() if hasattr(policy, 'global_step') else 0,
                        episode_reward=rewards,
                        episode_length=length,
                        episode_time=episode_time,
                        distance=info.get('distance'),
                        position_x=info.get('POSITION_X'),
                        position_y=info.get('POSITION_Y'),
                    )

                # Print episode summary
                if predictor is not None:
                    print("Episode finished. Sum of shaped rewards: {rewards:.2f}. "
                          "Length: {length}. Bonus: {life_bonus:.4f}.")
                    life_bonus = 0
                else:
                    print("Episode finished. Sum of shaped rewards: {rewards:.2f}. "
                          "Length: {length}.")

                if 'distance' in info:
                    print('Mario Distance Covered: {info["distance"]}')

                length = 0
                rewards = 0
                values = 0
                terminal_end = True
                last_features = policy.get_initial_features()  # reset lstm memory
                episode_start_time = time.time()

                # Reset environment
                if terminal or truncated or (timestep_limit and length >= timestep_limit):
                    last_state = env.reset()

            if info:
                # Summarize full game including all lives
                summary = tf.Summary()
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        summary.value.add(tag=k, simple_value=float(v))
                if terminal or truncated:
                    summary.value.add(tag='global/episode_value', simple_value=float(values))
                    values = 0
                    if predictor is not None:
                        summary.value.add(tag='global/episode_bonus', simple_value=float(ep_bonus))
                        ep_bonus = 0
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            if terminal_end:
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # Once we have enough experience, yield it
        yield rollout


class A3C:
    """Enhanced A3C implementation with modern logging and gymnasium support."""

    def __init__(
        self,
        env: Any,
        task: int,
        visualise: bool,
        unsup_type: Optional[str],
        env_wrap: bool = False,
        design_head: str = 'universe',
        no_reward: bool = False,
        logger: Optional[Logger] = None,
    ):
        """Initialize A3C trainer.

        Args:
            env: Environment to train on
            task: Task index for distributed training
            visualise: Whether to visualize
            unsup_type: Type of unsupervised learning ('action', 'state', 'stateAenc', or None)
            env_wrap: Whether environment is wrapped
            design_head: Network head design ('universe', 'nips', 'nature', 'doom')
            no_reward: Whether to remove rewards
            logger: Logger for metrics
        """
        self.task = task
        self.unsup = unsup_type is not None
        self.env_wrap = env_wrap
        self.env = env
        self.logger = logger

        predictor = None
        num_action = env.action_space.n
        worker_device = "/job:worker/task:{task}/cpu:0"

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, num_action, design_head)
                self.global_step = tf.get_variable(
                    "global_step",
                    [],
                    tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False
                )
                if self.unsup:
                    with tf.variable_scope("predictor"):
                        if 'state' in unsup_type:
                            self.ap_network = StatePredictor(
                                env.observation_space.shape, num_action, design_head, unsup_type
                            )
                        else:
                            self.ap_network = StateActionPredictor(
                                env.observation_space.shape, num_action, design_head
                            )

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, num_action, design_head)
                pi.global_step = self.global_step
                if self.unsup:
                    with tf.variable_scope("predictor"):
                        if 'state' in unsup_type:
                            self.local_ap_network = predictor = StatePredictor(
                                env.observation_space.shape, num_action, design_head, unsup_type
                            )
                        else:
                            self.local_ap_network = predictor = StateActionPredictor(
                                env.observation_space.shape, num_action, design_head
                            )

            # Computing A3C loss: https://arxiv.org/abs/1506.02438
            self.ac = tf.placeholder(tf.float32, [None, num_action], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # 1) The "policy gradients" loss: its derivative is precisely the policy gradient
            # Notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = -tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)  # Eq (19)

            # 2) Loss of value function: l2_loss = (x-y)^2/2
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))  # Eq (28)

            # 3) Entropy to ensure randomness
            entropy = -tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))

            # Final A3C loss: lr of critic is half of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy * constants['ENTROPY_BETA']

            # Compute gradients
            grads = tf.gradients(self.loss * 20.0, pi.var_list)  # batchsize=20

            # Computing predictor loss
            if self.unsup:
                if 'state' in unsup_type:
                    self.pred_loss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
                else:
                    self.pred_loss = constants['PREDICTION_LR_SCALE'] * (
                        predictor.invloss * (1 - constants['FORWARD_LOSS_WT']) +
                        predictor.forwardloss * constants['FORWARD_LOSS_WT']
                    )
                pred_grads = tf.gradients(self.pred_loss * 20.0, predictor.var_list)

                # Do not backprop to policy
                if constants['POLICY_NO_BACKPROP_STEPS'] > 0:
                    grads = [
                        tf.scalar_mul(
                            tf.to_float(tf.greater(self.global_step, constants['POLICY_NO_BACKPROP_STEPS'])),
                            grads_i
                        ) for grads_i in grads
                    ]

            self.runner = RunnerThread(
                env, pi, constants['ROLLOUT_MAXLEN'], visualise,
                predictor, env_wrap, no_reward, logger
            )

            # Storing summaries
            bs = tf.to_float(tf.shape(pi.x)[0])
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss)
                tf.summary.scalar("model/value_loss", vf_loss)
                tf.summary.scalar("model/entropy", entropy)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                if self.unsup:
                    tf.summary.scalar("model/pred_loss", self.pred_loss)
                    if 'action' in unsup_type:
                        tf.summary.scalar("model/inv_loss", predictor.invloss)
                        tf.summary.scalar("model/forward_loss", predictor.forwardloss)
                    tf.summary.scalar("model/pred_grad_global_norm", tf.global_norm(pred_grads))
                    tf.summary.scalar("model/pred_var_global_norm", tf.global_norm(predictor.var_list))
                self.summary_op = tf.summary.merge_all()
            else:
                # Legacy TensorFlow API
                tf.scalar_summary("model/policy_loss", pi_loss)
                tf.scalar_summary("model/value_loss", vf_loss)
                tf.scalar_summary("model/entropy", entropy)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                if self.unsup:
                    tf.scalar_summary("model/pred_loss", self.pred_loss)
                    if 'action' in unsup_type:
                        tf.scalar_summary("model/inv_loss", predictor.invloss)
                        tf.scalar_summary("model/forward_loss", predictor.forwardloss)
                    tf.scalar_summary("model/pred_grad_global_norm", tf.global_norm(pred_grads))
                    tf.scalar_summary("model/pred_var_global_norm", tf.global_norm(predictor.var_list))
                self.summary_op = tf.merge_all_summaries()

            # Clip gradients
            grads, _ = tf.clip_by_global_norm(grads, constants['GRAD_NORM_CLIP'])
            grads_and_vars = list(zip(grads, self.network.var_list))
            if self.unsup:
                pred_grads, _ = tf.clip_by_global_norm(pred_grads, constants['GRAD_NORM_CLIP'])
                pred_grads_and_vars = list(zip(pred_grads, self.ap_network.var_list))
                grads_and_vars = grads_and_vars + pred_grads_and_vars

            # Update global step by batch size
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # Each worker has a different set of adam optimizer parameters
            print("Optimizer: ADAM with lr: {constants['LEARNING_RATE']}")
            print("Input observation shape: {env.observation_space.shape}")
            opt = tf.train.AdamOptimizer(constants['LEARNING_RATE'])
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            # Copy weights from the parameter server to the local model
            sync_var_list = [v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]
            if self.unsup:
                sync_var_list += [v1.assign(v2) for v1, v2 in zip(predictor.var_list, self.ap_network.var_list)]
            self.sync = tf.group(*sync_var_list)

            # Initialize extras
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess: tf.Session, summary_writer: Any) -> None:
        """Start the A3C trainer.

        Args:
            sess: TensorFlow session
            summary_writer: Summary writer for TensorBoard
        """
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self) -> PartialRollout:
        """Take a rollout from the queue of the thread runner.

        Returns:
            Rollout from the queue
        """
        # Get top rollout from queue (FIFO)
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                # Now, get remaining *available* rollouts from queue and append them
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess: tf.Session) -> None:
        """Process a rollout and update parameters.

        Args:
            sess: TensorFlow session
        """
        sess.run(self.sync)  # Copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(
            rollout,
            gamma=constants['GAMMA'],
            lambda_=constants['LAMBDA'],
            clip=self.env_wrap
        )

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }
        if self.unsup:
            feed_dict[self.local_network.x] = batch.si[:-1]
            feed_dict[self.local_ap_network.s1] = batch.si[:-1]
            feed_dict[self.local_ap_network.s2] = batch.si[1:]
            feed_dict[self.local_ap_network.asample] = batch.a

        fetched = sess.run(fetches, feed_dict=feed_dict)
        if batch.terminal:
            print("Global Step Counter: {fetched[-1]}")

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

            # Log to Weights & Biases if available
            if self.logger:
                # Extract metrics from summary
                summary_proto = tf.Summary.FromString(fetched[0])
                metrics = {}
                for value in summary_proto.value:
                    metrics[value.tag] = value.simple_value

                self.logger.log_scalars(metrics, step=fetched[-1])

        self.local_steps += 1
