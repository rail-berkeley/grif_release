"""
Converts data from the BridgeData numpy format to TFRecord format.

Consider the following directory structure for the input data:

    bridgedata_numpy/
        rss/
            toykitchen2/
                set_table/
                    00/
                        train/
                            out.npy
                        val/
                            out.npy
        icra/
            ...

The --depth parameter controls how much of the data to process at the 
--input_path; for example, if --depth=5, then --input_path should be 
"bridgedata_numpy", and all data will be processed. If --depth=3, then 
--input_path should be "bridgedata_numpy/rss/toykitchen2", and only data 
under "toykitchen2" will be processed.

The same directory structure will be replicated under --output_path.  For 
example, in the second case, the output will be written to 
"{output_path}/set_table/00/...".

Can read/write directly from/to Google Cloud Storage.

Written by Kevin Black (kvablack@berkeley.edu).
"""
from absl import app, flags, logging
import numpy as np
import os
import tqdm
import tensorflow as tf
from multiprocessing import Pool, Manager
from jaxrl_m.data.language import lang_encode, load_mapping, flush_mapping
from jaxrl_m.vision.clip import create_from_checkpoint, process_image, process_text

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse. Looks for {input_path}/dir_1/dir_2/.../dir_{depth-1}/train/out.npy",
)
flags.DEFINE_bool("overwrite", True, "Overwrite existing files")
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_string(
    "model_ckpt", None, "Path to model checkpoint for writing embeddings"
)


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def process(path):
    global clip

    with tf.io.gfile.GFile(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    dirname = os.path.dirname(os.path.abspath(path))
    outpath = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-FLAGS.depth :])
    outpath = f"{outpath}/out.tfrecord"

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

    if len(arr) == 0:
        logging.info(f"Skipping {path}, empty")
        return

    tf.io.gfile.makedirs(os.path.dirname(outpath))

    # get rid of the confidence scores
    def clean_lang(lang):
        if lang is None:
            lang = ""
        lang = lang.strip()
        lines = lang.split("\n")
        lines = [l for l in lines if not "confidence" in l]
        # print("\n".join(lines))
        return "\n".join(lines)

    with tf.io.TFRecordWriter(outpath) as writer:
        text_batch = []
        img_batch = []
        if FLAGS.model_ckpt:
            for traj in arr:
                text = traj["language"][0]
                text = clean_lang(text)
                if text is None:
                    text = "placeholder"
                else:
                    text = text.split("\n")[0]  # multiple labels just take first

                s0 = traj["observations"][0]["images0"]
                g = traj["observations"][-1]["images0"]
                s0 = process_image(s0)
                g = process_image(g)
                img = np.concatenate([s0, g], axis=-1)

                text_batch.append(text)
                img_batch.append(img)
            text_batch = process_text(text_batch)
            img_batch = np.concatenate(img_batch, axis=0)
            clip_output = clip(pixel_values=img_batch, **text_batch)
            text_embeds = clip_output["text_embeds"]
            image_embeds = clip_output["image_embeds"]

        for i, traj in enumerate(arr):
            if FLAGS.model_ckpt:
                traj_text_embed = text_embeds[i : i + 1]
                traj_text_embed = np.repeat(
                    traj_text_embed, len(traj["actions"]), axis=0
                )
                traj_image_embed = image_embeds[i : i + 1]
                traj_image_embed = np.repeat(
                    traj_image_embed, len(traj["actions"]), axis=0
                )

            with lock:
                encoded_language = tensor_feature(
                    np.array(
                        [
                            lang_encode(clean_lang(x) if x else None)
                            for x in traj["language"]
                        ]
                    )
                )
            truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
            truncates[-1] = True
            feature = {
                "observations/images0": tensor_feature(
                    np.array(
                        [o["images0"] for o in traj["observations"]],
                        dtype=np.uint8,
                    )
                ),
                "observations/state": tensor_feature(
                    np.array(
                        [o["state"] for o in traj["observations"]],
                        dtype=np.float32,
                    )
                ),
                "next_observations/images0": tensor_feature(
                    np.array(
                        [o["images0"] for o in traj["next_observations"]],
                        dtype=np.uint8,
                    )
                ),
                "next_observations/state": tensor_feature(
                    np.array(
                        [o["state"] for o in traj["next_observations"]],
                        dtype=np.float32,
                    )
                ),
                "actions": tensor_feature(np.array(traj["actions"], dtype=np.float32)),
                "terminals": tensor_feature(
                    np.zeros(len(traj["actions"]), dtype=np.bool_)
                ),
                "truncates": tensor_feature(truncates),
                "language": encoded_language,
            }
            if FLAGS.model_ckpt:
                feature["text_embed"] = tensor_feature(traj_text_embed)
                feature["image_embed"] = tensor_feature(traj_image_embed)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def main(_):
    global clip
    if FLAGS.model_ckpt:
        clip = create_from_checkpoint(FLAGS.model_ckpt)

    global lock
    assert FLAGS.depth >= 1
    manager = Manager()

    lock = manager.Lock()
    flush_mapping(FLAGS.output_path)
    load_mapping(FLAGS.output_path, manager.dict)
    paths = tf.io.gfile.glob(
        tf.io.gfile.join(FLAGS.input_path, *(["*?"] * (FLAGS.depth - 1)))
    )
    code_path = tf.io.gfile.join(FLAGS.output_path, "language_encodings.json")
    paths = [os.path.join(p, "train/out.npy") for p in paths] + [
        os.path.join(p, "val/out.npy") for p in paths
    ]
    # with Pool(FLAGS.num_workers) as p:
    # list(tqdm.tqdm(p.imap(process, paths), total=len(paths)))
    error_paths = []
    for path in tqdm.tqdm(paths):
        #try:
        process(path)
        # break
        #except Exception as e:
        #    error_paths.append(path)
        #    print("Error on path", path)

    flush_mapping(FLAGS.output_path)
    print(error_paths)
    with tf.io.gfile.GFile(
        os.path.join(FLAGS.output_path, "error_paths.txt"), "w"
    ) as f:
        f.write("\n".join(error_paths))


if __name__ == "__main__":
    app.run(main)
