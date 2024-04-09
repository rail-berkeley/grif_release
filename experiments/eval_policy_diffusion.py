import functools
import sys
from widowx_envs.widowx.widowx_env import (
    BridgeDataRailRLPrivateWidowX,
    BridgeDataRailRLPrivateVRWidowX,
)
import os
import numpy as np
from PIL import Image
from flax.training import checkpoints
from jaxrl_m.agents.continuous.gc_iql import create_iql_learner
from jaxrl_m.agents.continuous.gc_bc import create_bc_learner
import traceback
import wandb
from jaxrl_m.vision import encoders as vision_encoders
from absl import app, flags, logging
import time
from datetime import datetime
import jax
import time
from denoising_diffusion_flax.model import EmaTrainState, create_model_def
from denoising_diffusion_flax import utils
from denoising_diffusion_flax.sampling import sample_loop
import ml_collections

np.set_printoptions(suppress=True)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "policy_checkpoint", None, "Path to policy checkpoint", required=True
)
flags.DEFINE_string(
    "diffusion_checkpoint", None, "Path to diffusion checkpoint", required=True
)
flags.DEFINE_string(
    "policy_wandb", None, "Policy checkpoint wandb run name", required=True
)
flags.DEFINE_string(
    "diffusion_wandb", None, "Diffusion checkpoint wandb run name", required=True
)

flags.DEFINE_integer(
    "diffusion_sample_steps", 200, "Number of timesteps to use for diffusion sampler"
)
flags.DEFINE_float("diffusion_eta", 0.0, "Eta to use for diffusion sampler")
flags.DEFINE_float("diffusion_w", 1.0, "CFG weight to use for diffusion sampler")

flags.DEFINE_string("video_save_path", None, "Path to save video")

flags.DEFINE_integer("num_timesteps", 50, "Number of timesteps per subgoal")
flags.DEFINE_integer("num_subgoals", 5, "Number of subgoals to run")

flags.DEFINE_bool("blocking", True, "Use the blocking controller")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2

# mutable global gripper state to implement sticky gripper
IS_GRIPPER_CLOSED = False
NUM_CONSECUTIVE_GRIPPER_CHANGE_ACTIONS = 0

FIXED_STD = np.array(
    [0.1159527, 0.057708, 0.09830328, 0.10394894, 0.20113328, 0.11955251, 0.0001]
)


def unnormalize_action(action, mean, std):
    return action * std + mean


def load_policy_checkpoint():
    # load information from wandb
    api = wandb.Api()
    run = api.run(FLAGS.policy_wandb)
    model_config = run.config["model_config"]
    model_constructor = run.config["model_constructor"]
    action_mean = np.array(run.config["bridgedata_config"]["action_metadata"]["mean"])
    action_std = np.array(run.config["bridgedata_config"]["action_metadata"]["std"])

    # create agent
    if "iql" in model_constructor:
        model_config["agent_kwargs"]["policy_kwargs"]["fixed_std"] = FIXED_STD
    elif "bc" in model_constructor:
        model_config["agent_kwargs"]["policy_kwargs"]["fixed_std"] = FIXED_STD
    else:
        raise ValueError(f"Unknown model constructor {model_constructor}")
    model_config["agent_kwargs"]["optim_kwargs"] = {"learning_rate": 0.0}

    encoder_def = vision_encoders[model_config["encoder"]](
        **model_config["encoder_kwargs"]
    )
    agent = globals()[model_constructor](
        seed=0,
        encoder_def=encoder_def,
        observations=np.zeros((128, 128, 3), dtype=np.uint8),
        goals=np.zeros((128, 128, 3), dtype=np.uint8),
        actions=np.zeros(7, dtype=np.float32),
        **model_config["agent_kwargs"],
    )
    params = checkpoints.restore_checkpoint(FLAGS.policy_checkpoint, target=None)[
        "model"
    ]["params"]
    agent = agent.replace(model=agent.model.replace(params=params))

    return agent, action_mean, action_std


def load_diffusion_checkpoint():
    # load config from wandb
    api = wandb.Api()
    run = api.run(FLAGS.diffusion_wandb)
    config = ml_collections.ConfigDict(run.config["config"])

    # create model def
    model_def = create_model_def(
        **config.model,
    )

    # load weights
    ckpt_dict = checkpoints.restore_checkpoint(FLAGS.diffusion_checkpoint, target=None)
    state = EmaTrainState(
        step=0,
        apply_fn=model_def.apply,
        params=ckpt_dict["params"],
        params_ema=ckpt_dict["params_ema"],
        tx=None,
        opt_state=None,
    )

    # parse ddpm params
    ddpm_params = utils.get_ddpm_params(config.ddpm)

    # compile sample loop
    if FLAGS.diffusion_sample_steps is None:
        num_timesteps = config.ddpm.timesteps
    else:
        num_timesteps = FLAGS.diffusion_sample_steps
    sample_loop_jit = jax.jit(
        functools.partial(
            sample_loop,
            ddpm_params=ddpm_params,
            num_timesteps=num_timesteps,
            eta=FLAGS.diffusion_eta,
            w=FLAGS.diffusion_w,
            self_condition=config.ddpm.self_condition,
        )
    )

    return state, sample_loop_jit


def rollout_subgoal(
    rng, env, agent, goal_image, action_mean, action_std, num_timesteps
):
    goal_obs = {
        "image": goal_image,
    }

    global IS_GRIPPER_CLOSED
    global NUM_CONSECUTIVE_GRIPPER_CHANGE_ACTIONS

    # env.reset()
    # env.start()
    obs = env._get_obs()
    last_tstep = time.time()
    images = []
    t = 0
    try:
        while t < num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                image_obs = (
                    obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                obs = {"image": image_obs, "proprio": obs["state"]}

                last_tstep = time.time()

                rng, key = jax.random.split(rng)
                action = np.array(agent.sample_actions(obs, goal_obs, seed=key))
                action = unnormalize_action(action, action_mean, action_std)

                # sticky gripper logic
                if (action[-1] < 0.5) != IS_GRIPPER_CLOSED:
                    NUM_CONSECUTIVE_GRIPPER_CHANGE_ACTIONS += 1
                else:
                    NUM_CONSECUTIVE_GRIPPER_CHANGE_ACTIONS = 0

                if NUM_CONSECUTIVE_GRIPPER_CHANGE_ACTIONS >= STICKY_GRIPPER_NUM_STEPS:
                    IS_GRIPPER_CLOSED = not IS_GRIPPER_CLOSED
                    NUM_CONSECUTIVE_GRIPPER_CHANGE_ACTIONS = 0

                action[-1] = 0.0 if IS_GRIPPER_CLOSED else 1.0

                ### Preprocess action ###
                if NO_PITCH_ROLL:
                    action[3] = 0
                    action[4] = 0
                if NO_YAW:
                    action[5] = 0

                ### Env step ###
                obs, rew, done, info = env.step(
                    action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                )

                image_formatted = np.concatenate((goal_image, image_obs), axis=0)
                images.append(Image.fromarray(image_formatted))

                t += 1
    except Exception as e:
        logging.error(traceback.format_exc())
        return images, False

    return images, True


def main(_):
    assert os.path.exists(FLAGS.policy_checkpoint)
    assert os.path.exists(FLAGS.diffusion_checkpoint)

    # load policy checkpoint
    agent, action_mean, action_std = load_policy_checkpoint()

    # load diffusion checkpoint
    diffusion_state, sample_loop_jit = load_diffusion_checkpoint()

    ### Setup env ###
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.23, -0.15, 0, -1.57, 0],
            [0.36, 0.23, 0.18, 1.57, 0],
        ],  # (0.03 for lower z) #[[0.17, -0.08, 0.03, -1.57, 0], [0.35, 0.08, 0.1,  1.57, 0]],
        # 'action_clipping': 'xyz',
        "action_clipping": None,
        "catch_environment_except": False,
        "start_state": None,
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    env.reset()
    env.start()
    rng = jax.random.PRNGKey(0)
    images = []
    for i in range(FLAGS.num_subgoals):
        # get current image and normalize to [-1, 1]
        obs = env._get_obs()
        image_obs = obs["image"].reshape(3, 128, 128).transpose(1, 2, 0) * 2 - 1

        # sample from diffusion model
        logging.info(f"Subgoal {i}: sampling from diffusion model...")
        rng, key = jax.random.split(rng)
        image_goal = np.squeeze(sample_loop_jit(key, diffusion_state, image_obs[None]))
        image_goal = np.clip(image_goal * 127.5 + 127.5 + 0.5, 0, 255).astype(np.uint8)

        # rollout subgoal
        logging.info(f"Subgoal {i}: rolling out...")
        rng, key = jax.random.split(rng)
        new_images, success = rollout_subgoal(
            key, env, agent, image_goal, action_mean, action_std, FLAGS.num_timesteps
        )
        images += new_images
        if not success:
            break

    # Save Video
    if FLAGS.video_save_path is not None:
        os.makedirs(FLAGS.video_save_path, exist_ok=True)
        save_path = os.path.join(
            FLAGS.video_save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.gif")
        )
        logging.info(f"Saving Video at {save_path}...")
        images[0].save(
            save_path,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=200,
            loop=0,
        )


if __name__ == "__main__":
    app.run(main)
