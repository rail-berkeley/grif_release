from functools import partial
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.common.common import shard_batch
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
from jaxrl_m.utils.train_utils import load_recorded_video
from jaxrl_m.utils.sim_utils import make_mujoco_gc_env
from jaxrl_m.common.evaluation import evaluate_gc, supply_rng
import tensorflow as tf
import tqdm
import jax
import jax.numpy as jnp
from absl import app, flags, logging
from ml_collections import config_flags
import numpy as np
from flax.training import checkpoints

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_sim",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # load action metadata
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.config.data_path, "train/metadata.npy"), "rb"
    ) as f:
        action_metadata = np.load(f, allow_pickle=True).item()

    # load eval goals
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.config.data_path, "val/eval_goals.npy"), "rb"
    ) as f:
        eval_goals = np.load(f, allow_pickle=True).item()

    # create sim environment
    eval_env = make_mujoco_gc_env(
        env_name=FLAGS.config.env_name,
        max_episode_steps=FLAGS.config.max_episode_steps,
        action_metadata=action_metadata,
        save_video=FLAGS.config.save_video,
        save_video_dir=tf.io.gfile.join(save_dir, "videos"),
        save_video_prefix="eval",
        goals=eval_goals,
    )

    # load datasets
    train_paths = tf.io.gfile.glob(f"{FLAGS.config.data_path}/train/*.tfrecord")
    val_paths = tf.io.gfile.glob(f"{FLAGS.config.data_path}/val/*.tfrecord")
    train_data = BridgeDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        num_devices=num_devices,
        train=True,
        action_metadata=action_metadata,
        relabel_actions=False,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = BridgeDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        action_metadata=action_metadata,
        relabel_actions=False,
        train=False,
        **FLAGS.config.dataset_kwargs,
    )
    train_data_iter = train_data.get_iterator()

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding)

    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )
    if resume_path := FLAGS.config.get("resume_path", None) is not None:
        agent = checkpoints.restore_checkpoint(resume_path, target=agent)
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = shard_batch(next(train_data_iter), sharding)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if i % FLAGS.config.eval_interval == 0:
            logging.info("Validation...")
            timer.tick("val")
            metrics = []
            i = 0
            for batch in val_data.get_iterator():
                metrics.append(agent.get_debug_metrics(batch))
                i += 1
                if i >= FLAGS.config.num_val_batches:
                    break
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            metrics["log_probs"] *= -1
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

            rng, policy_key = jax.random.split(rng)
            policy_fn = supply_rng(
                partial(agent.sample_actions, argmax=FLAGS.config.deterministic_eval),
                rng=policy_key,
            )

            logging.info("Evaluating...")
            timer.tick("evaluation")
            eval_env.goal_sampler = eval_goals
            eval_env.start_recording(
                FLAGS.config.num_episodes_per_video, FLAGS.config.num_episodes_per_row
            )
            eval_info = evaluate_gc(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.config.eval_episodes,
                return_trajectories=False,
            )
            wandb_logger.log({f"evaluation": eval_info}, step=i)
            if FLAGS.config.save_video:
                eval_video = load_recorded_video(video_path=eval_env.current_save_path)
                wandb_logger.log({"evaluation/video": eval_video}, step=i)
            timer.tock("evaluation")

        if i % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if i % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)

            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
