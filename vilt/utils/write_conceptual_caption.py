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


def write_arrow(examples, sink, dataset_root):
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
    return []


def dump_arrow(
    batch_idx, dataset_root, caption_paths, img_to_captions, batch_size, split
):
    with pa.OSFile(
        f"{dataset_root}/conceptual_caption_{split}_{batch_idx}.arrow", "wb"
    ) as sink:
        examples = list()
        for idx, path in enumerate(
            tqdm(
                caption_paths[batch_idx : batch_idx + batch_size],
                position=batch_idx % (batch_size * 4),
            )
        ):
            binary = open(path, "rb").read()
            img_id = os.path.basename(path)
            captions = img_to_captions[img_id]
            examples.append([binary, captions, img_id, split])
            if not idx % 20000:
                examples = write_arrow(examples, sink, dataset_root)

        write_arrow(examples, sink, dataset_root)


def make_arrow(root, dataset_root, annotations_json, splits=[""], batch_size=50000):
    dataset = json.load(open(annotations_json, "rb"))["images"]
    paths = list(glob(os.path.join(root, "images/*")))
    random.shuffle(paths)

    for split in splits:
        iid2captions = dict()
        dataset_split = [img for img in dataset if img["split"] == split]
        for annot in tqdm(dataset_split):
            iid2captions[annot["filename"]] = [
                sentence["raw"] for sentence in annot["sentences"]
            ]
        caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

        load_data = partial(
            dump_arrow,
            dataset_root=dataset_root,
            caption_paths=caption_paths,
            img_to_captions=iid2captions,
            batch_size=batch_size,
            split=split,
        )

        print(f"Dumping batch {split}")
        batched_annots = process_map(
            load_data,
            [idxs for idxs in range(0, len(caption_paths), batch_size)],
            max_workers=4,
        )


if __name__ == "__main__":
    make_arrow(
        "/data/datasets/conceptual_captions",
        "/data/jaredfer/vilt/datasets/arrows_pretrain",
        "/data/datasets/conceptual_captions/annotations/dataset_cc.json",
        splits=["train"],
        batch_size=100000,
    )
