import gc
import json
import multiprocessing
import os
import random
from functools import partial
from glob import glob
from tqdm.contrib.concurrent import process_map

import pandas as pd
import pyarrow as pa
from tqdm import tqdm


def dump_arrow(batch_idx, dataset_root, caption_paths, img_to_captions, batch_size):
    with pa.OSFile(f"{dataset_root}/sbu_{batch_idx}.arrow", "wb") as sink:
        examples = list()
        for idx, path in enumerate(
            tqdm(
                caption_paths[batch_idx : batch_idx + batch_size],
                position=batch_idx % (4 * batch_size),
            )
        ):
            binary = open(path, "rb").read()
            img_id = os.path.basename(path)
            captions = img_to_captions[img_id]
            examples.append([binary, captions, img_id, "train"])

            if not idx % 20000:
                dataframe = pd.DataFrame(
                    examples, columns=["image", "caption", "image_id", "split"]
                )
                table = pa.Table.from_pandas(dataframe)
                os.makedirs(dataset_root, exist_ok=True)
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)

                del dataframe
                del table
                del examples
                gc.collect()

                examples = list()
        dataframe = pd.DataFrame(
            examples, columns=["image", "caption", "image_id", "split"]
        )
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

        del dataframe
        del table
        del examples
        gc.collect()


def make_arrow(root, dataset_root, annotations_json, splits=[""], batch_size=50000):
    dataset = json.load(open(annotations_json, "rb"))
    paths = list(glob(os.path.join(root, "images/*")))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in dataset]

    load_data = partial(
        dump_arrow,
        dataset_root=dataset_root,
        caption_paths=caption_paths,
        img_to_captions=dataset,
        batch_size=batch_size,
    )

    batched_annots = process_map(
        load_data,
        [idxs for idxs in range(0, len(caption_paths), batch_size)],
        max_workers=4,
    )


if __name__ == "__main__":
    make_arrow(
        "/data/datasets/SBU",
        "/data/jaredfer/vilt/datasets/arrows_pretrain",
        "/data/datasets/SBU/annot.json",
        batch_size=100000,
    )
