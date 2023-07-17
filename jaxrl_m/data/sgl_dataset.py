from typing import Iterator, List, Union, Optional, Iterable
import tensorflow as tf
import os
from jaxrl_m.data.bridge_dataset import BridgeDataset


# this is for the manually collected validation sets
class SGLDataset(BridgeDataset):
    def __init__(
        self,
        root_data_path: str,
    ):
        tf_paths = tf.io.gfile.glob(os.path.join(root_data_path, "data.tfrecord"))
        caption_path = tf_paths[0].replace("data.tfrecord", "captions.txt")

        # read captions
        with tf.io.gfile.GFile(caption_path, "r") as f:
            content = f.read()
            lines = content.split("\n")
            captions = [line.split(",")[2].strip() for line in lines]
        self.captions = captions 
        dataset = self._construct_tf_dataset(tf_paths)
        dataset = dataset.batch(100, drop_remainder=False)
        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths):
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(
            self._decode_example_sgl, num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset

    PROTO_TYPE_SPEC = {
        "start_frame": tf.uint8,
        "stop_frame": tf.uint8,
        "label": tf.uint32,
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

        return {
            "observations": {
                "image": parsed_tensors["start_frame"],
            },
            "next_observations": {
                "image": parsed_tensors["stop_frame"],
            },
            "goals": {
                "image": parsed_tensors["stop_frame"],
                "language": parsed_tensors["label"],
            },
        }

    def decode_lang(self, i):
        return self.captions[i]


if __name__ == "__main__":
    dataset = SGLDataset("gs://rail-tpus-andre/bridge_validation/scene1")
    data_iter = dataset.get_iterator()
    batch = next(data_iter)
    print(batch)
