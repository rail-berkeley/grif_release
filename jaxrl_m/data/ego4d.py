import tensorflow as tf


PROTO_TYPE_SPEC = {"images": tf.string, "text": tf.string}


def get_ego4d_dataloader(
        path, 
        batch_size, 
        shuffle_buffer_size=25000, 
        cache=False, 
        take_every_n=1, 
        val_split=0.1,
        seed=42,
        train_file_idxs=None,
        val_file_idxs=None, 
    ):

    # get the tfrecord files
    dataset = tf.data.Dataset.list_files(tf.io.gfile.join(path, "*.tfrecord"))

    # at this point the cardinality is still known, so we split it into train and val
    num_files = dataset.cardinality().numpy()
    num_val_files = int(num_files * val_split)
    num_train_files = num_files - num_val_files

    # shuffle the dataset
    dataset = dataset.shuffle(num_files, seed=seed)
    datasets = {}
    if train_file_idxs is None:
        # split into train and val at the file level
        datasets["train"] = dataset.take(num_train_files)
        datasets["val"] = dataset.skip(num_train_files)
    else:
        train_filter_func = lambda i, data: tf.reduce_any(tf.math.equal(tf.math.mod(i, 64), train_file_idxs))
        datasets["train"] = dataset.enumerate().filter(train_filter_func).map(lambda i, data: data)
        val_filter_func = lambda i, data: tf.reduce_any(tf.math.equal(tf.math.mod(i, 64), val_file_idxs))
        datasets["val"] = dataset.enumerate().filter(val_filter_func).map(lambda i, data: data)

    for split, dataset in datasets.items(): 
        # read them
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # decode the examples (yields videos)
        dataset = dataset.map(_decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # cache all the dataloading
        if cache:
            dataset = dataset.cache()

        # add goals (yields videos)
        dataset = dataset.map(_add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to get individual frames
        dataset = dataset.unbatch()

        if take_every_n > 1:
            dataset = dataset.shard(take_every_n, index=0)
        

        # process each frame
        dataset = dataset.map(_process_frame, num_parallel_calls=tf.data.AUTOTUNE)

        # shuffle the dataset
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

        # batch the dataset
        dataset = dataset.batch(
            batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True
        )

        # restructure the batches
        dataset = dataset.map(_restructure_batch, num_parallel_calls=tf.data.AUTOTUNE)

        # always prefetch last
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # repeat the dataset
        if split == "train":
            dataset = dataset.repeat()

        datasets[split] = dataset

    return datasets

def _restructure_batch(batch):
    # batch is a dict with keys "images", "text", "goals", and "frame_indices"
    # "images" is a tensor of shape [batch_size, 224, 224, 3]
    # "text" is a tensor of shape [batch_size]
    # "goals" is a tensor of shape [batch_size]
    
   return {
        "observations": {
            "image": batch["images"],
        },
        "goals": {
            "image": batch["goals"],
            "language": batch["text"],
        },
        "actions": tf.zeros([tf.shape(batch["images"])[0], 10], dtype=tf.float32),
   }


def _decode_example(example_proto):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string) for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in PROTO_TYPE_SPEC.items()
    }

    return parsed_tensors


def _add_goals(video):
    # video is a dict with keys "images" and "text"
    # "images" is a tensor of shape [n_frames, 224, 224, 3]
    # "text" is a tensor of shape [n_frames]

    # for now: for frame i, select a goal uniformly from the range [i, n_frames)
    num_frames = tf.shape(video["images"])[0]
    rand = tf.random.uniform(shape=[num_frames], minval=0, maxval=1, dtype=tf.float32)
    offsets = tf.cast(
        tf.floor(rand * tf.cast(tf.range(num_frames)[::-1], tf.float32)), tf.int32
    )
    indices = tf.range(num_frames) + offsets
    video["goals"] = tf.gather(video["images"], indices)    

    # for now: just get rid of text
    video["text"] = tf.tile(tf.expand_dims(video["text"], 0), [num_frames])

    return video


def _process_frame(frame):
    for key in ["images", "goals"]:
        frame[key] = tf.io.decode_jpeg(frame[key])
        # this will throw an error if any images aren't 224x224x3
        frame[key] = tf.ensure_shape(frame[key], [224, 224, 3])
        # may want to think more carefully about the resize method
        # frame[key] = tf.image.resize(frame[key], [128, 128], method="lanczos3")
        # normalize to [-1, 1]
        # frame[key] = (frame[key] / 127.5) - 1
        # convert to float32
        # frame[key] = tf.cast(frame[key], tf.float32)

    return frame


if __name__ == "__main__":
    import tqdm

    datasets = get_ego4d_dataloader(
        "gs://rail-tpus-kevin/ego4d-tfrecord", 256, 100, cache=False, take_every_n=50
    )


    with tqdm.tqdm() as pbar:
        for batch in datasets['train']:
            pbar.update(1)
            print(batch["goals"]["language"])
