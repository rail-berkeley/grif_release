from typing import Iterator, List, Union, Optional, Iterable

import tensorflow as tf
import os
from jaxrl_m.data.bridge_dataset import BridgeDataset

EXCLUDE_KEYWORDS = ["Pretending", "Failing"]
INCLUDE_TASKS_ = [
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Taking [something] out of [something]",
    "Pulling [something] from right to left",
    "Pushing [something] off of [something]",
    "Moving [something] down",
    "Pulling [something] out of [something]",
    "Pushing [something] from left to right",
    "Moving [something] closer to [something]",
    "Opening [something]",
    "Pulling [something] from left to right",
    "Moving [something] and [something] away from each other",
    "Putting [something] behind [something]",
    "Pushing [something] onto [something]",
]

INCLUDE_TASKS = [
    task.replace("[", "").replace("]", "").replace(" ", "_") for task in INCLUDE_TASKS_
]


class SS2Dataset(BridgeDataset):
    """
    Dataloader for Something-Something V2 dataset.
    Imitating BridgeDataset.
    """

    def __init__(
        self,
        root_data_path: str,
        seed: int,
        batch_size: int,
        shuffle_buffer_size: int = 10000,
        prefetch_num_batches: int = 5,
        train: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: Optional[dict] = None,
        clip_preprocessing: bool = False,
    ):
        self.augment_kwargs = augment_kwargs or {}
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently

        # sub_folders = tf.io.gfile.glob(os.path.join(root_data_path, "*"))
        # sub_folders = [
        #     sub_folder
        #     for sub_folder in sub_folders
        #     if not any([keyword in sub_folder for keyword in EXCLUDE_KEYWORDS])
        # ]
        # data_paths = [
        #     tf.io.gfile.glob(os.path.join(sub_folder, "*.tfrecord"))
        #     for sub_folder in sub_folders
        # ]

        # datasets = []
        # sizes = []
        # for sub_data_paths in data_paths:
        #     sub_data_paths = [p for p in sub_data_paths if tf.io.gfile.exists(p)]
        #     if len(sub_data_paths) == 0:
        #         continue
        #     print(f"Found {len(sub_data_paths)} tfrecords in {sub_data_paths[0]}")
        #     datasets.append(self._construct_tf_dataset(sub_data_paths, seed))
        #     sizes.append(len(sub_data_paths))

        data_paths = tf.io.gfile.glob(os.path.join(root_data_path, "*/*.tfrecord"))
        data_paths = [
            data_path
            for data_path in data_paths
            if not any([keyword in data_path for keyword in EXCLUDE_KEYWORDS])
        ]
        data_paths = [
            data_path
            for data_path in data_paths
            if any([task in data_path for task in INCLUDE_TASKS])
        ]

        # shuffle data paths
        data_paths = tf.random.shuffle(data_paths, seed=seed).numpy().tolist()
        print(f"Found {len(data_paths)} tfrecords in {root_data_path}")
        datasets = [self._construct_tf_dataset(data_paths, seed)]
        sizes = [len(data_paths)]

        total_size = sum(sizes)
        sample_weights = [size / total_size for size in sizes]

        if train:
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .repeat()
                    .shuffle(
                        max(1, int(shuffle_buffer_size * sample_weights[i])), seed + i
                    )
                )
        else:
            # TODO ?
            for i in range(len(datasets)):
                datasets[i] = datasets[i].shuffle(
                    max(1, int(shuffle_buffer_size * sample_weights[i])), seed + i
                )

        dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            sample_weights,
            seed=seed,
            stop_on_empty_dataset=train,
        )

        if train and augment:
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        if clip_preprocessing:
            dataset = dataset.map(
                self._clip_preprocess, num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.prefetch(prefetch_num_batches)
        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(
            len(paths), seed=seed
        )
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(
            self._decode_example_sgl, num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset

    PROTO_TYPE_SPEC = {
        "frames": tf.uint8,
        "task_id": tf.uint32,
    }

    def _decode_example_sgl(self, example_proto):
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }

        # only care about initial and final frames
        # TODO add goal labeling scheme
        # TODO could simplify bridge dataset this way; probably reduces duplicates too
        return {
            "observations": {
                "image": parsed_tensors["frames"][0],
            },
            "next_observations": {
                "image": parsed_tensors["frames"][-1],
            },
            "goals": {
                "image": parsed_tensors["frames"][-1],
                "language": parsed_tensors["task_id"][0],
            },
            "initial_obs": {
                "image": parsed_tensors["frames"][0],
            },
        }


if __name__ == "__main__":
    root_data_path = "gs://rail-tpus-andre/something-something/tf_fixed/train/"
    seed = 42
    batch_size = 64
    shuffle_buffer_size = 10

    dataset = SS2Dataset(
        root_data_path=root_data_path,
        seed=seed,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        augment=False,
        # augment_kwargs=
    )

    data_iter = dataset.get_iterator()
    batch = next(data_iter)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(batch["observations"]["image"][0])
    plt.savefig("obs.png")
    print(batch)

    labels_path = "gs://rail-tpus-andre/something-something/tf/labels/"
    from jaxrl_m.data.ss2_language import load_mapping, get_encodings

    load_mapping(labels_path)
    lang_to_code, code_to_lang = get_encodings()

    print([(k, v) for k, v in lang_to_code.items()][:10])
    print([(k, v) for k, v in code_to_lang.items()][:10])
