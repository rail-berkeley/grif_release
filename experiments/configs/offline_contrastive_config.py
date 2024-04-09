import ml_collections
import flax.linen as nn

multimodal_config_proto = dict(
    device="gpu",  # tpu
    resume_path=None,
    resume_clip_parts_path=None,
    resume_step=None,
    seed=42,
    num_steps=5000,
    log_interval=25,
    eval_interval=100,
    save_interval=100,
    save_dir="gs://rail-tpus-andre/logs",
    dataset="bridgedata",  # "ego4d" or "bridgedata"
    lang_encoder=dict(
        # type="muse",
        type="clip",
        clip_variant="openai/clip-vit-base-patch32",
        # type="pretrained",
        # name="distilbert-base-uncased",
        kwargs=dict(mlp_kwargs=None, freeze_encoder=False),
    ),
    image_encoder=dict(
        type="clip",  # "encoders",
        clip_variant="openai/clip-vit-base-patch32",
        clip_use_pretrained_params=True,
        # name="resnetv1-34-bridge",
        # kwargs=dict(
        #    pooling_method="avg",
        #    add_spatial_coordinates=True,
        #    act="swish",
        # ),
    ),
    agent_kwargs=dict(
        learning_rate=3e-5,
        text_learning_rate=3e-6,
        warmup_steps=2000,
        decay_steps=int(2e6),
        dropout_rate=0.0,
        mlp_kwargs=None,  # dict(
        # hidden_dims=(512, ), #512, 512),
        # activation=nn.relu,
        # activate_final=False,
        # ),
    ),
    ego4d_kwargs=dict(
        path="gs://rail-tpus-kevin/ego4d-tfrecord",
        batch_size=64,
        shuffle_buffer_size=10,
        take_every_n=50,
        cache=False,
        seed=32,
        train_file_idxs=list(range(32)),
        val_file_idxs=list(range(32, 35)),
    ),
    bridge_data_path="gs://rail-tpus-andre/new_tf",
    bridge_batch_size=256,
    bridge_val_batch_size=64,
    bridge_split_strategy="task",
    bridge_split_prop=0.2,
    bridge_augment_tasks=True,
    dataset_kwargs=dict(
        shuffle_buffer_size=25000,
        labeled_ony=True,
        simple_goal=True,
        prefetch_num_batches=20,
        relabel_actions=True,
        goal_relabel_reached_proportion=0.0,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    ),
    ss2_train_path="gs://rail-tpus-andre/something-something/tf_fixed/train/",
    ss2_val_path="gs://rail-tpus-andre/something-something/tf_fixed/validation/",
    ss2_labels_path="gs://rail-tpus-andre/something-something/labels/",
    ss2_batch_size=256,
    ss2_val_batch_size=256,
    ss2_dataset_kwargs=dict(
        shuffle_buffer_size=25000,
        prefetch_num_batches=20,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    ),
    val_set_path="gs://rail-tpus-andre/bridge_validation/scene1",
    # "gs://rail-tpus-andre/bridge_validation/scene2",
    # "gs://rail-tpus-andre/bridge_validation/scene3",
    # "gs://rail-tpus-andre/bridge_validation/scene4",
    # "gs://rail-tpus-andre/bridge_validation/transfer",
)


def update_config(proto=multimodal_config_proto, **kwargs):
    result = dict(proto).copy()
    for key, value in kwargs.items():
        if type(result.get(key)) == dict:
            value = dict(update_config(proto=result[key], **kwargs[key]))
        result[key] = value
    return ml_collections.ConfigDict(result)


def get_config(config_string):
    possible_structures = {
        "lcbc": update_config(
            task_encoders=dict(language="resnetv1-18-bridge-task"),
        ),
        "sg_sl": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge", language="resnetv1-18-bridge-task"
            ),
        ),
        "sg_sl_align": update_config(
            task_encoders=dict(
                image="resnetv1-18-bridge", language="resnetv1-18-bridge-task"
            ),
            agent_kwargs=dict(
                alignment=1.0,
            ),
        ),
        "tiny": update_config(
            task_encoders=dict(image="resnetv1-18-bridge", language=""),
            agent_kwargs=dict(
                alignment=1.0,
            ),
            pretrained_encoder="distilbert-base-uncased",
        ),
        "contrastive_tpu": update_config(
            device="tpu",
        ),
        "contrastive_gpu": update_config(
            device="gpu",
            bridge_batch_size=64,
            bridge_val_batch_size=64,
            dataset_kwargs=dict(
                shuffle_buffer_size=500,
            ),
        ),
    }

    possible_structures["contrastive_tpu_muse"] = update_config(
        possible_structures["contrastive_tpu"],
        lang_encoder=dict(type="muse"),
    )

    possible_structures["contrastive_tpu_resnet_muse"] = update_config(
        possible_structures["contrastive_tpu"],
        lang_encoder=dict(type="muse"),
        image_encoder=dict(
            type="encoders",
            name="resnetv1-18-bridge",
            kwargs=dict(
                pooling_method="avg",
                add_spatial_coordinates=True,
                act="swish",
            ),
            mlp_kwargs=dict(
                hidden_dims=(512, 512),
                activation=nn.relu,
                activate_final=False,
            ),
        ),
    )

    possible_structures["contrastive_tpu_resnet_clip_lang"] = update_config(
        possible_structures["contrastive_tpu"],
        lang_encoder=dict(
            type="clip",
            clip_variant="openai/clip-vit-base-patch32",
            kwargs=dict(mlp_kwargs=None, freeze_encoder=True),
        ),
        image_encoder=dict(
            type="encoders",
            name="resnetv1-18-bridge",
            kwargs=dict(
                pooling_method="avg",
                add_spatial_coordinates=True,
                act="swish",
            ),
            mlp_kwargs=dict(
                hidden_dims=(512, 512),
                activation=nn.relu,
                activate_final=False,
            ),
        ),
    )

    return possible_structures[config_string]
