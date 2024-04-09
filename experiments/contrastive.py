from functools import partial
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.common.common import shard_batch
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list
from jaxrl_m.data.ss2 import SS2Dataset
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
import tensorflow as tf

from jaxrl_m.data.ego4d import get_ego4d_dataloader
from transformers import AutoTokenizer, FlaxAutoModel
from jaxrl_m.common.common import JaxRLTrainState, nonpytree_field
from flax.core import FrozenDict
from jaxrl_m.common.typing import PRNGKey

import tqdm
import jax
import jax.numpy as jnp
from jaxrl_m.data.language import load_mapping, lang_encodings, lang_decode
from jaxrl_m.data.ss2_language import (
    load_mapping as load_ss2_mapping,
    lang_decode as lang_decode_ss2,
)
from jaxrl_m.data.sgl_dataset import SGLDataset
from absl import app, flags, logging
from ml_collections import config_flags
import numpy as np
from flax.training import checkpoints
import os
import flax.linen as nn
from jaxrl_m.common.common import MLP
import flax
from flax import traverse_util
import optax
from flax.training.common_utils import onehot
from flax.core.frozen_dict import freeze
import pickle

from transformers import FlaxCLIPModel, FlaxCLIPTextModel  # , FlaxCLIPVisionModel
from transformers import CLIPProcessor

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import functools
import wandb

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("mode", "train", "script mode")  # train, eval_checkpoint, debug

flags.DEFINE_string("dataset_", None, "dataset")
flags.DEFINE_string("resume_path_", None, "resume_path")
flags.DEFINE_integer("resume_step_", None, "resume_step")
flags.DEFINE_string("bridge_split_strategy_", None, "split by task or traj")
flags.DEFINE_string("val_set_path_", None, "val set path")
flags.DEFINE_bool("bridge_augment_tasks_", False, "Use gpt augmented tasks")
flags.DEFINE_bool("incr_order", False, "experimental feature")
flags.DEFINE_float("p_unshuffle", 0.0, "-")

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

device_count = None


def recursive_concat(batch1, batch2):
    if isinstance(batch1, dict):
        return {k: recursive_concat(batch1[k], batch2[k]) for k in batch1}
    elif isinstance(batch1, list):
        return [recursive_concat(b1, b2) for b1, b2 in zip(batch1, batch2)]
    else:
        return jnp.concatenate([batch1, batch2], axis=0)


class PretrainedLangEncoder(nn.Module):
    pretrained_lang_encoder: nn.Module
    mlp_kwargs: Dict = None
    freeze_encoder: bool = True
    projection_dim: int = None

    @nn.compact
    def __call__(self, inputs):
        output = self.pretrained_lang_encoder(**inputs).pooler_output
        if self.freeze_encoder:
            output = jax.lax.stop_gradient(output)
        if self.mlp_kwargs is not None:
            output = MLP(**self.mlp_kwargs)(output)
        if self.projection_dim is not None:
            output = nn.Dense(
                self.projection_dim, name="text_projection", use_bias=False
            )(output)
        return output

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        pretrained_encoder = FlaxAutoModel.from_pretrained(pretrained_name_or_path)
        pretrained_encoder_def = pretrained_encoder.module
        pretrained_encoder_params = pretrained_encoder.params
        return (
            cls(pretrained_encoder_def, **kwargs),
            pretrained_encoder_params,
        )


class PretrainedImageEncoder(nn.Module):
    # this key is used to load the pretrained params
    pretrained_image_encoder: nn.Module
    projection_dim: int = None

    @nn.compact
    def __call__(self, inputs):
        x = self.pretrained_image_encoder(inputs).pooler_output
        if self.projection_dim is not None:
            x = nn.Dense(self.projection_dim, name="image_projection", use_bias=False)(
                x
            )
        return x


class ContrastiveModule(nn.Module):
    encoders: Dict[str, nn.Module]
    temperature_init: float = 0.1
    mlp_kwargs: Dict[str, Any] = None
    dropout_rate: float = 0.1

    def setup(self):
        self.temperature = self.param(
            "temperature", nn.initializers.constant(jnp.log(0.07)), ()
        )
        if self.mlp_kwargs is not None:
            self.mlps = dict(
                image=MLP(**self.mlp_kwargs),
                language=MLP(**self.mlp_kwargs),
            )
        else:
            self.mlps = dict(
                image=lambda x: x,
                language=lambda x: x,
            )

    # mode = 'image' or 'language'
    def encode(
        self, input_data: jnp.ndarray, mode: str = "image", training: bool = False
    ) -> jnp.ndarray:
        x = self.encoders[mode](input_data)
        if mode == "language":
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return self.mlps[mode](x)

    @nn.compact
    def __call__(
        self,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        mode: str = "compute_loss",
        training: bool = False,
        return_logps: bool = False,
    ) -> jnp.ndarray:
        image_batch = batch["image_batch"]
        text_batch = batch["text_batch"]
        # img_unique_mask = batch['img_unique_mask']
        # lang_unique_mask = batch['lang_unique_mask']
        lang_dupe_matrix = batch["lang_dupe_matrix"]

        if mode == "image":
            return self.encode(image_batch, mode=mode)
        elif mode == "language":
            return self.encode(text_batch, mode=mode)

        image_embeddings = self.encode(image_batch, mode="image", training=training)
        text_embeddings = self.encode(text_batch, mode="language", training=training)

        image_embeddings = image_embeddings / jnp.linalg.norm(
            image_embeddings, axis=-1, keepdims=True
        )
        text_embeddings = text_embeddings / jnp.linalg.norm(
            text_embeddings, axis=-1, keepdims=True
        )

        # Clip the temperature
        temperature = jnp.clip(self.temperature, jnp.log(0.01), jnp.log(100))

        logits = jnp.matmul(image_embeddings, text_embeddings.T) / jnp.exp(temperature)
        if mode == "logits":
            return {
                "logits": logits,
                "image_embeddings": image_embeddings,
                "text_embeddings": text_embeddings,
            }

        img2text_logps = nn.log_softmax(logits, axis=1)
        text2img_logps = nn.log_softmax(logits, axis=0)

        # collect the logprobs of the correct img-text pairs (not just diag cuz duplicates)
        img2text_pos_logps = img2text_logps - 1e9 * (1 - lang_dupe_matrix)
        text2img_pos_logps = text2img_logps - 1e9 * (1 - lang_dupe_matrix)

        img2text_loss = -jnp.mean(
            jax.scipy.special.logsumexp(img2text_pos_logps, axis=1)
        )
        text2img_loss = -jnp.mean(
            jax.scipy.special.logsumexp(text2img_pos_logps, axis=0)
        )
        loss = (img2text_loss + text2img_loss) / 2

        img2text_correct_pr = jnp.mean(
            jnp.sum(jnp.exp(img2text_logps) * lang_dupe_matrix, axis=1)
        )
        text2img_correct_pr = jnp.mean(
            jnp.sum(jnp.exp(text2img_logps) * lang_dupe_matrix, axis=0)
        )

        # accuracy of retrieving one of the correct labels
        img2text_accuracy = jnp.mean(
            jnp.max(img2text_logps, axis=1) == jnp.max(img2text_pos_logps, axis=1)
        )
        text2img_accuracy = jnp.mean(
            jnp.max(text2img_logps, axis=0) == jnp.max(text2img_pos_logps, axis=0)
        )

        # top 5 accuracy
        img2text_top5_accuracy = jnp.mean(
            jnp.max(img2text_pos_logps, axis=1)
            >= jnp.sort(img2text_logps, axis=1)[:, -5]
        )
        text2img_top5_accuracy = jnp.mean(
            jnp.max(text2img_pos_logps, axis=0)
            >= jnp.sort(text2img_logps, axis=0)[-5, :]
        )

        update_info = {
            "loss": loss,
            "img2text_loss": img2text_loss,
            "text2img_loss": text2img_loss,
            "img2text_correct_pr": img2text_correct_pr,
            "text2img_correct_pr": text2img_correct_pr,
            "img2text_accuracy": img2text_accuracy,
            "text2img_accuracy": text2img_accuracy,
            "img2text_top5_accuracy": img2text_top5_accuracy,
            "text2img_top5_accuracy": text2img_top5_accuracy,
            "temperature": temperature,
            "num_unique_langs": batch["num_unique_langs"][0],
            # "lang_ids": batch["lang_ids"],
        }

        if FLAGS.mode == "debug":
            update_info.update(
                {
                    "lang_dupe_matrix": lang_dupe_matrix,
                    "text2img_logps": text2img_logps,
                    "img2text_logps": img2text_logps,
                    "text2img_diag_logps": jnp.diag(text2img_logps),
                    "text2img_diag_prs": jnp.exp(jnp.diag(text2img_logps)),
                }
            )

        if return_logps:
            update_info.update(
                {
                    "lang_dupe_matrix": lang_dupe_matrix,
                    "img2text_logps": img2text_logps,
                    "text2img_logps": text2img_logps,
                }
            )

        return loss, update_info


class ContrastiveAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()

    @jax.jit
    def update(self, batch: FrozenDict, pmap_axis: str = None):
        def loss_fn(params, rng):
            return self.state.apply_fn(
                params, batch, mode="compute_loss", rngs={"dropout": rng}, training=True
            )

        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @functools.partial(jax.jit, static_argnames=("return_logps",))
    def metrics(self, batch: FrozenDict, pmap_axis: str = None, return_logps=False):
        _, info = self.state.apply_fn(
            self.state.params,
            batch,
            mode="compute_loss",
            training=False,
            return_logps=return_logps,
        )
        return info

    def get_logits(self, batch: FrozenDict, pmap_axis: str = None):
        return self.state.apply_fn(self.state.params, batch, mode="logits")

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        batch: FrozenDict,
        encoders: Dict[str, nn.Module],
        mlp_kwargs: Dict[str, Any] = None,
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        pretrained_params: Dict[str, Any] = None,
        dropout_rate: float = 0.0,
        text_learning_rate: float = 3e-4,
    ):
        model_def = ContrastiveModule(
            encoders=encoders, mlp_kwargs=mlp_kwargs, dropout_rate=dropout_rate
        )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

        text_lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=text_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, batch, training=False)

        def find_and_replace(params, key, replacement):
            for k in params.keys():
                if k == key:
                    params[k] = replacement
                    logging.info(f"replaced {key} in params")
                    return
                if isinstance(params[k], type(params)):
                    find_and_replace(params[k], key, replacement)

        params = params.unfreeze()
        for key in pretrained_params:
            find_and_replace(params, key, pretrained_params[key])
        params = freeze(params)

        # image_tx = optax.adamw(learning_rate=lr_schedule)
        # text_tx = optax.adamw(learning_rate=text_lr_schedule)
        # partition_optimizers = {"text": text_tx, "image": image_tx}
        # param_partitions = freeze(
        #    traverse_util.path_aware_map(
        #        lambda path, v: "text"
        #        if "text_projection" in path or "pretrained_lang_encoder" in path
        #        else "image",
        #        params,
        #    )
        # )
        # tx = optax.multi_transform(partition_optimizers, param_partitions)
        tx = optax.adamw(learning_rate=lr_schedule)

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )
        return cls(state=state, lr_schedule=lr_schedule)


def preprocess_batch(batch, processors, multi_label_rng, val_set=None):
    if val_set is None:
        if FLAGS.config.dataset == "ego4d":
            sentences = batch["goals"]["language"]
            lang_dupe_matrix = jnp.eye(len(sentences))
        elif FLAGS.config.dataset == "bridgedata":
            lang_ids = batch["goals"]["language"]
            sentences = [lang_decode(lang_id, multi_label_rng) for lang_id in lang_ids]
            _, lang_unique_idxs = jnp.unique(lang_ids, return_index=True)
            lang_dupe_matrix = lang_ids[:, None] == lang_ids[None, :]
        elif FLAGS.config.dataset == "ss2":
            lang_ids = batch["goals"]["language"]
            sentences = [lang_decode_ss2(lang_id) for lang_id in lang_ids]
            lang_dupe_matrix = jnp.eye(len(sentences))
    else:
        lang_ids = batch["goals"]["language"]
        sentences = [val_set.decode_lang(lang_id.item()) for lang_id in lang_ids]
        lang_dupe_matrix = jnp.eye(len(sentences))

    if isinstance(sentences[0], bytes):
        sentences = [s.decode("utf-8") for s in sentences]

    if FLAGS.config.lang_encoder.type == "pretrained":
        inputs = processors["tokenizer"](
            sentences, return_tensors="jax", padding=True, truncation=True
        )
        inputs = {k: jnp.array(v) for k, v in inputs.items()}
    elif FLAGS.config.lang_encoder.type == "muse":
        inputs = [jnp.array(processors["multi_embed"](s)) for s in sentences]
        inputs = jnp.concatenate(inputs, axis=0)
    elif FLAGS.config.lang_encoder.type == "clip":
        inputs = processors["clip_processor"](
            sentences, return_tensors="np", padding=True
        )
        inputs = {k: jnp.array(v) for k, v in inputs.items()}
        inputs["position_ids"] = jnp.expand_dims(
            jnp.arange(inputs["input_ids"].shape[1]), axis=0
        ).repeat(inputs["input_ids"].shape[0], axis=0)

    s0 = batch["observations"]["image"]
    if s0.shape[-1] == 4:
        s0 = s0[:, :, :, :3]
    g = batch["goals"]["image"]
    if g.shape[-1] == 4:
        g = g[:, :, :, :3]

    if FLAGS.config.image_encoder.type == "clip":
        s0 = processors["clip_processor"](images=s0, return_tensors="np", padding=True)[
            "pixel_values"
        ]
        s0 = s0.transpose((0, 2, 3, 1))
        g = processors["clip_processor"](images=g, return_tensors="np", padding=True)[
            "pixel_values"
        ]
        g = g.transpose((0, 2, 3, 1))
    elif FLAGS.config.image_encoder.type == "encoders":
        s0 = (s0 / 127.5) - 1.0
        g = (g / 127.5) - 1.0

    image_batch = jnp.concatenate((s0, g), axis=-1)

    if FLAGS.config.device == "tpu":
        if len(image_batch) % device_count != 0:
            truncated_batch_size = len(image_batch) // device_count * device_count
            image_batch = image_batch[:truncated_batch_size]
            if (
                FLAGS.config.lang_encoder.type == "pretrained"
                or FLAGS.config.lang_encoder.type == "clip"
            ):
                inputs = {k: v[:truncated_batch_size] for k, v in inputs.items()}
            elif FLAGS.config.lang_encoder.type == "muse":
                inputs = inputs[:truncated_batch_size]
            lang_dupe_matrix = lang_dupe_matrix[
                :truncated_batch_size, :truncated_batch_size
            ]

    if val_set or FLAGS.config.dataset == "ego4d" or FLAGS.config.dataset == "ss2":
        return (
            FrozenDict(
                image_batch=image_batch,
                text_batch=inputs,
                lang_dupe_matrix=lang_dupe_matrix,
                num_unique_langs=jnp.ones(image_batch.shape[0]) * len(sentences),
            ),
            sentences,
        )
    elif FLAGS.config.dataset == "bridgedata":
        return (
            FrozenDict(
                image_batch=image_batch,
                text_batch=inputs,
                lang_dupe_matrix=lang_dupe_matrix,
                num_unique_langs=jnp.ones(image_batch.shape[0]) * len(lang_unique_idxs),
                # lang_ids=jnp.array(lang_ids),
            ),
            sentences,
        )


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    global device_count
    device_count = num_devices
    logging.info(f"devices: {devices}")
    timer = Timer()

    for flag in FLAGS:
        if flag.endswith("_"):
            if getattr(FLAGS, flag) is None or getattr(FLAGS, flag) == "":
                continue
            assert flag[:-1] in FLAGS.config, f"override flag {flag} not in config"
            FLAGS.config[flag[:-1]] = getattr(FLAGS, flag)

    # prevent tensorflow from using GPUs
    if FLAGS.config.device == "tpu":
        tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_bridgedata",
            "exp_descriptor": FLAGS.name,
        },
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.mode == "debug",
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # load models and preprocessors
    processors = {}
    pretrained_params = {}
    assert FLAGS.config.lang_encoder.type in ["pretrained", "muse", "clip"]
    if FLAGS.config.lang_encoder.type == "pretrained":
        lang_encoder_def, lang_encoder_params = PretrainedLangEncoder.from_pretrained(
            FLAGS.config.lang_encoder.name, **FLAGS.config.lang_encoder.kwargs
        )
        pretrained_params["pretrained_lang_encoder"] = lang_encoder_params
        processors["tokenizer"] = AutoTokenizer.from_pretrained(
            FLAGS.config.lang_encoder.name
        )
    elif FLAGS.config.lang_encoder.type == "muse":
        lang_encoder_def = lambda x: x
        MULTI_MODULE = (
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        )
        muse_model = hub.load(MULTI_MODULE)

        @functools.lru_cache(maxsize=None)
        def multi_embed(x):
            with tf.device("/cpu:0"):
                return muse_model(x).numpy()

        processors["multi_embed"] = multi_embed
    elif FLAGS.config.lang_encoder.type == "clip":
        clip = FlaxCLIPModel.from_pretrained(FLAGS.config.lang_encoder.clip_variant)
        clip, clip_variables = clip.module, {"params": clip.params}
        text_model, text_model_vars = clip.bind(clip_variables).text_model.unbind()

        # text_model = FlaxCLIPTextModel.from_pretrained(
        #     FLAGS.config.lang_encoder.clip_variant
        # )
        # text_model, text_model_vars = text_model.module, {"params": text_model.params}

        processors["clip_processor"] = CLIPProcessor.from_pretrained(
            FLAGS.config.image_encoder.clip_variant
        )
        lang_encoder_def = PretrainedLangEncoder(
            text_model, projection_dim=512, **FLAGS.config.lang_encoder.kwargs
        )
        pretrained_params["pretrained_lang_encoder"] = text_model_vars["params"]
        pretrained_params["text_projection"] = clip_variables["params"][
            "text_projection"
        ]

    assert FLAGS.config.image_encoder.type in ["encoders", "clip"]
    if FLAGS.config.image_encoder.type == "encoders":
        image_encoder_def = encoders[FLAGS.config.image_encoder.name](
            **FLAGS.config.image_encoder.kwargs
        )
    elif FLAGS.config.image_encoder.type == "clip":
        assert FLAGS.config.lang_encoder.type == "clip"
        vision_model, vision_model_vars = clip.bind(
            clip_variables
        ).vision_model.unbind()
        image_encoder_def = PretrainedImageEncoder(vision_model, projection_dim=512)

        if FLAGS.config.image_encoder.clip_use_pretrained_params:
            # if we want to use pretrained params, need to modify shape for 6-channel input
            pek_key = "embeddings/patch_embedding/kernel".split("/")
            vision_model_vars = vision_model_vars.unfreeze()
            pek_params = vision_model_vars["params"]["embeddings"]["patch_embedding"][
                "kernel"
            ]
            sg_pek_params = jnp.concatenate([pek_params, pek_params], axis=2) / 2.0
            vision_model_vars["params"]["embeddings"]["patch_embedding"][
                "kernel"
            ] = sg_pek_params
            vision_model_vars = freeze(vision_model_vars)
            pretrained_params["pretrained_image_encoder"] = vision_model_vars["params"]
            pretrained_params["image_projection"] = clip_variables["params"][
                "visual_projection"
            ]

    if FLAGS.config.bridge_augment_tasks:
        multi_label_rng = np.random.RandomState(FLAGS.config.seed)
    else:
        multi_label_rng = None
    process_batch_train = functools.partial(
        preprocess_batch, processors=processors, multi_label_rng=multi_label_rng
    )
    process_batch_val = functools.partial(
        preprocess_batch, processors=processors, multi_label_rng=None
    )

    # load dataset
    assert FLAGS.config.dataset in ["ego4d", "bridgedata", "ss2"]
    if FLAGS.config.dataset == "ego4d":
        datasets = get_ego4d_dataloader(
            **FLAGS.config.ego4d_kwargs,
        )
        train_data = datasets["train"]
        val_data = datasets["val"]

        train_data_iter = map(
            process_batch_train, map(FrozenDict, train_data.as_numpy_iterator())
        )
        get_val_data_iters = {
            "validation": lambda: map(
                process_batch_val, map(FrozenDict, val_data.as_numpy_iterator())
            )
        }

    elif FLAGS.config.dataset == "bridgedata":
        load_mapping(
            FLAGS.config.bridge_data_path,
            augmented=FLAGS.config.bridge_augment_tasks,
        )

        train_task_paths = [
            glob_to_path_list(
                path,
                prefix=FLAGS.config.bridge_data_path,
                exclude=FLAGS.bridgedata_config.exclude,
            )
            for path in FLAGS.bridgedata_config.include
        ]
        val_task_paths = [
            glob_to_path_list(
                path,
                prefix=FLAGS.config.bridge_data_path,
                exclude=FLAGS.bridgedata_config.exclude,
            )
            for path in FLAGS.bridgedata_config.include
        ]

        train_paths = [
            [os.path.join(path, "train/out.tfrecord") for path in sub_list]
            for sub_list in train_task_paths
        ]
        val_paths = [
            [os.path.join(path, "val/out.tfrecord") for path in sub_list]
            for sub_list in val_task_paths
        ]

        if FLAGS.config.bridge_split_strategy == "task":
            # lang_ids = list(lang_encodings().keys())
            # lang_ids = sorted(lang_ids)
            pass
            # nothing to do here

        elif FLAGS.config.bridge_split_strategy == "traj":
            raise NotImplementedError  # not maintained
            val_lang_ids = None
            train_lang_ids = None

        # since we split by language, merge all paths into one list
        train_data = BridgeDataset(
            train_paths,
            FLAGS.config.seed,
            batch_size=FLAGS.config.bridge_batch_size // 2,
            num_devices=num_devices,
            train=True,
            action_metadata=FLAGS.bridgedata_config.action_metadata,
            sample_weights=FLAGS.bridgedata_config.sample_weights,
            **FLAGS.config.dataset_kwargs,
        )
        train_data_shuffle_iter = map(process_batch_train, train_data.get_iterator())

        # this will serve as a harder training set for contrastive learning
        # cuz trajectories from the same scene are likely to be in the same batches
        unshuffled_kwargs = FLAGS.config.dataset_kwargs
        unshuffled_kwargs["shuffle_buffer_size"] = 1
        train_data_unshuffled = BridgeDataset(
            train_paths,
            FLAGS.config.seed,
            batch_size=FLAGS.config.bridge_batch_size // 2,
            num_devices=num_devices,
            train=True,
            action_metadata=FLAGS.bridgedata_config.action_metadata,
            sample_weights=FLAGS.bridgedata_config.sample_weights,
            **unshuffled_kwargs,
        )
        train_data_unshuffled_iter = map(
            process_batch_train, train_data_unshuffled.get_iterator()
        )

        def mixed_train_iter():
            while True:
                batch_shuffled, sents_shuffled = next(train_data_shuffle_iter)
                batch_unshuffled, sents_unshuffled = next(train_data_unshuffled_iter)
                # recursively concat the two batches
                batch = recursive_concat(batch_shuffled, batch_unshuffled)
                batch = FrozenDict(batch)
                sents = sents_shuffled + sents_unshuffled
                yield batch, sents

        train_data_iter = mixed_train_iter()

        # We need to do this for the validation order to be the same across epochs
        get_val_data_iters = {
            "validation": lambda: map(
                process_batch_val,
                BridgeDataset(
                    val_paths,
                    FLAGS.config.seed,
                    batch_size=FLAGS.config.bridge_val_batch_size,
                    action_metadata=FLAGS.bridgedata_config.action_metadata,
                    train=False,
                    **FLAGS.config.dataset_kwargs,
                ).get_iterator(),
            ),
        }

    elif FLAGS.config.dataset == "ss2":
        load_ss2_mapping(FLAGS.config.ss2_labels_path)

        train_data = SS2Dataset(
            root_data_path=FLAGS.config.ss2_train_path,
            seed=FLAGS.config.seed,
            batch_size=FLAGS.config.ss2_batch_size,
            train=True,
            **FLAGS.config.ss2_dataset_kwargs,
        )
        train_data_iter = map(process_batch_train, train_data.get_iterator())

        get_val_data_iters = {
            "validation": lambda: map(
                process_batch_val,
                SS2Dataset(
                    root_data_path=FLAGS.config.ss2_val_path,
                    seed=FLAGS.config.seed,
                    batch_size=FLAGS.config.ss2_val_batch_size,
                    train=False,
                    **FLAGS.config.ss2_dataset_kwargs,
                ).get_iterator(),
            ),
        }

    # throw in the manual validation set
    # each should be a single batch
    if FLAGS.mode == "eval_checkpoint":
        get_val_data_iters = {}
        val_set_path = FLAGS.config.val_set_path
        val_set_name = "valset_" + val_set_path.split("/")[-1]
        val_set = SGLDataset(val_set_path)
        val_set_preprocess_batch = functools.partial(
            preprocess_batch,
            processors=processors,
            multi_label_rng=multi_label_rng,
            val_set=val_set,
        )
        get_val_data_iters[val_set_name] = lambda: map(
            val_set_preprocess_batch, val_set.get_iterator()
        )

    example_batch, example_sents = next(train_data_iter)
    image_batch = example_batch["image_batch"]
    logging.info(f"Batch size: {image_batch.shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(f"Batch size per device: {image_batch.shape[0] // num_devices}")
    example_sents = "\n".join(example_sents[:5])
    logging.info(f"Example sentences: {example_sents}")

    if FLAGS.config.device == "tpu":
        # we shard the leading dimension (batch dimension) accross all devices evenly
        sharding = jax.sharding.PositionalSharding(devices)
        example_batch = shard_batch(example_batch, sharding)

    if FLAGS.config.resume_path:
        logging.info("Resuming from checkpoint, not loading pretrained params")
        pretrained_params = {}
    elif FLAGS.config.resume_clip_parts_path:
        # load pretrained clip parts from another checkpoint
        raw = checkpoints.restore_checkpoint(
            ckpt_dir=FLAGS.config.resume_clip_parts_path, target=None
        )
        logging.info("Loading parts from checkpoint, dump pretrained params")
        pretrained_params = {}
        params = raw["state"]["params"]["params"]
        pretrained_params["pretrained_image_encoder"] = params["encoders_image"][
            "pretrained_image_encoder"
        ]
        pretrained_params["image_projection"] = params["encoders_image"][
            "image_projection"
        ]
        pretrained_params["pretrained_lang_encoder"] = params["encoders_language"][
            "pretrained_lang_encoder"
        ]
        pretrained_params["text_projection"] = params["encoders_language"][
            "text_projection"
        ]

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = ContrastiveAgent.create(
        rng=construct_rng,
        batch=example_batch,
        encoders=dict(
            image=image_encoder_def,
            language=lang_encoder_def,
        ),
        # mlp_kwargs=dict(
        #    hidden_dims=(512, 512),
        #    activation=nn.relu,
        #    activate_final=False,
        # ),
        pretrained_params=pretrained_params,
        **FLAGS.config.agent_kwargs,
    )

    if FLAGS.config.resume_path:
        agent = checkpoints.restore_checkpoint(
            FLAGS.config.resume_path, target=agent, step=FLAGS.config.resume_step
        )

    if FLAGS.config.device == "tpu":
        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
        agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    def run_validation(val_data_iter, step, split_name, num_per_batch=5):
        timer.tick("val")
        metrics = []
        val_batch_infos = []
        for j, (batch, sents) in enumerate(tqdm.tqdm(val_data_iter)):
            if FLAGS.config.device == "tpu":
                batch = shard_batch(batch, sharding)
            update_info = agent.metrics(batch)
            update_info = jax.device_get(update_info)
            metrics.append(update_info)

            if j < 5:
                extra_update_info = agent.metrics(batch, return_logps=True)
                extra_update_info = jax.device_get(extra_update_info)
                batch = jax.device_get(batch)
                val_batch_infos.append((batch, sents, extra_update_info))

        metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
        wandb_logger.log({f"{split_name}": metrics}, step=step)
        timer.tock("val")

        timer.tick("val_log")
        img2text_table = wandb.Table(
            columns=["images", "caption", "top5_langs", "top5_prs"]
        )
        text2img_table = wandb.Table(
            columns=["caption", "images", "top5_images", "top5_prs"]
        )
        for batch, sents, update_info in val_batch_infos:
            # log the predicted text for the first 5 images in batch, for 5 batches
            for k in range(min(num_per_batch, batch["image_batch"].shape[0])):
                s0 = batch["image_batch"][k, :, :, :3]
                g = batch["image_batch"][k, :, :, 3:]

                # for the first table
                image_pair = np.concatenate([s0, g], axis=0)
                image_pair = wandb.Image(image_pair)
                caption = sents[k]
                img2text_logps = update_info["img2text_logps"][k]
                topk_lang_idxs = np.argsort(img2text_logps)[::-1][:5]
                topk_langs = "\n".join([sents[i] for i in topk_lang_idxs])
                topk_lang_prs = np.exp(img2text_logps[topk_lang_idxs])
                img2text_table.add_data(image_pair, caption, topk_langs, topk_lang_prs)

                # for the second table
                text2img_logps = update_info["text2img_logps"][:, k]
                topk_img_idxs = np.argsort(text2img_logps)[::-1][:5]
                topk_imgs = [
                    np.concatenate(
                        [
                            batch["image_batch"][i, :, :, :3],
                            batch["image_batch"][i, :, :, 3:],
                        ],
                        axis=0,
                    )
                    for i in topk_img_idxs
                ]
                topk_imgs = np.concatenate(topk_imgs, axis=1)
                topk_imgs = wandb.Image(topk_imgs)
                topk_img_prs = np.exp(text2img_logps[topk_img_idxs])
                text2img_table.add_data(caption, image_pair, topk_imgs, topk_img_prs)

        wandb.log(
            {
                f"{split_name}/img2text": img2text_table,
                f"{split_name}/text2img": text2img_table,
            },
            step=step,
        )
        timer.tock("val_log")

    # if FLAGS.mode == "debug":
    #     logging.info("Running debug...")
    #     train_task_freqs = np.zeros(len(lang_encodings()))
    #     for i in tqdm.tqdm(range(int(1e3))):
    #         batch, sents = next(train_data_iter)
    #         lang_ids = batch["lang_ids"]
    #         for lang_id in lang_ids:
    #             train_task_freqs[lang_id] += 1
    #     np.save("train_task_freqs.npy", train_task_freqs)
    #     logging.info("Saved train task freqs")

    #     val_task_freqs = np.zeros(len(lang_encodings()))
    #     for batch, sents in get_val_data_iter():
    #         lang_ids = batch["lang_ids"]
    #         for lang_id in lang_ids:
    #             val_task_freqs[lang_id] += 1
    #     np.save("val_task_freqs.npy", val_task_freqs)
    #     logging.info("Saved val task freqs")
    #     return

    if FLAGS.mode == "eval_checkpoint":
        logging.info("Evaluating checkpoint...")
        for split_name, get_val_data_iter in get_val_data_iters.items():
            run_validation(
                get_val_data_iter(),
                0,
                split_name,
                num_per_batch=100 if "valset" in split_name else 5,
            )
        return

    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch, _ = next(train_data_iter)
        if FLAGS.config.device == "tpu":
            batch = shard_batch(batch, sharding)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0:
            for split_name, get_val_data_iter in get_val_data_iters.items():
                run_validation(get_val_data_iter(), i + 1, split_name, num_per_batch=5)

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
