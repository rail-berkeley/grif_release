from absl import app, flags, logging
from PIL import Image
from multiprocessing import Pool, Manager

import tensorflow as tf
import numpy as np
import glob
import os
import json
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_path", "/nfs/kun2/users/andrehe/SS2", "Path to input directory"
)
flags.DEFINE_string(
    "labels_path", "/nfs/kun2/users/andrehe/SS2_labels", "Path to labels json files"
)
flags.DEFINE_string(
    "output_path",
    "gs://rail-tpus-andre/something-something/tf_fixed",
    "Path to output directory",
)
flags.DEFINE_bool("overwrite", True, "Overwrite existing files")
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("chunk_size", 100, "Number of videos per tfrecord")


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def process(args):
    video_infos, split, action_class = args

    outpath = os.path.join(FLAGS.output_path, split, action_class.replace(" ", "_"))
    # outpath = f"{outpath}/"

    if tf.io.gfile.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            try:
                tf.io.gfile.rmtree(outpath)
            except:
                pass
        else:
            logging.info(f"Skipping {outpath}")
            return

    tf.io.gfile.makedirs(os.path.dirname(outpath))

    for i in range(0, len(video_infos), FLAGS.chunk_size):
        chunk = video_infos[i : min(i + FLAGS.chunk_size, len(video_infos))]
        tf_path = os.path.join(outpath, f"{i // FLAGS.chunk_size}.tfrecord")
        with tf.io.TFRecordWriter(tf_path) as writer:
            for video_info in chunk:
                video_path = os.path.join(FLAGS.input_path, video_info["id"])

                frame_paths = tf.io.gfile.glob(os.path.join(video_path, "*.png"))
                if len(frame_paths) == 0:
                    logging.info(f"Skipping {video_path}, empty")
                    continue

                try:
                    # read frames
                    frames = [
                        np.array(Image.open(frame_path).resize((128, 128)))
                        for frame_path in frame_paths
                    ]
                except:
                    logging.info(f"Skipping {video_path}, error reading frames")
                    continue

                # use this to find caption
                video_id = int(video_info["id"])

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "frames": tensor_feature(np.array(frames, dtype=np.uint8)),
                            "task_id": tensor_feature(
                                (np.ones(len(frames)) * video_id).astype(np.uint32)
                            ),
                        }
                    )
                )
                writer.write(example.SerializeToString())
                logging.info(f"Processed {video_path}")


def main(_):
    with open(os.path.join(FLAGS.labels_path, "labels.json")) as f:
        labels = json.load(f)
    classes = labels.keys()

    map_inputs = []
    for split in ["train", "validation"]:
        # for split in ["validation"]:
        with open(os.path.join(FLAGS.labels_path, f"{split}.json")) as f:
            metadata = json.load(f)

        for action_class in classes:
            video_infos = [
                vid_info
                for vid_info in metadata
                if vid_info["template"].replace("[", "").replace("]", "")
                == action_class
            ]
            map_inputs.append((video_infos, split, action_class))

    with Pool(FLAGS.num_workers) as p:
        list(
            tqdm.tqdm(
                p.imap(process, map_inputs),
                total=len(map_inputs),
            )
        )


if __name__ == "__main__":
    app.run(main)
