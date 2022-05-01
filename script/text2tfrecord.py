"""tokenization to bpe or character embeddings of text datasets"""

import argparse
import io
import multiprocessing
import os
import shutil
import time

import jsonlines
import requests
import simdjson
import tensorflow as tf
import zstandard
from google.cloud import storage
from transformers import GPT2TokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="text",
                    help="Name of output files will be name_i.tfrecords where i is the number of the file")
parser.add_argument("--procs", type=int, default=2, help="Number of processes in multiprocessing")
parser.add_argument("--output_dir", type=str, default="gs://homebrewnlp-eu/the-token-pile/",
                    help="Where to put tfrecords (in a bucket)")
parser.add_argument("--int64", type=bool, default=True, help="Whether to encode as bytes or int64")
parser.add_argument("--buffer_size", type=int, default=2 ** 29, help="This is a minimum size, not a maximum size. "
                                                                     "tfrecords will have this minimum size as well.")
parser.add_argument("--separator", type=str, default=chr(4),
                    help="separator to place between files in chunk mode."
                         "Default is \x04 (chr(4)) in case of byte encodings, "
                         "but should be changed to <|endoftext|> for BPE")


def file_generator(args, pid, procs):
    base_url = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
    splits = 30
    parse_fn = simdjson.Parser().parse
    tmp_name = f".tmp.download.{pid}"

    def _json_parser(x):
        return parse_fn(x.encode()).as_dict()

    for i in range(pid, splits, procs):
        with requests.get(base_url.replace("%s", str(i).zfill(2)), stream=True) as r, open(tmp_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        with open(tmp_name, 'rb') as f:
            for item in jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)),
                                         loads=_json_parser):
                if isinstance(item, dict):
                    item = item['text']
                if isinstance(item, list):
                    item = args.separator.join(item)
                yield item
        os.remove(tmp_name)


def create_tfrecords(args, pid, procs):
    slash_idx = args.output_dir.find('/')
    bucket_name, output_dir = args.output_dir[:slash_idx], args.output_dir[slash_idx + 1:]
    bucket = storage.Client().get_bucket(bucket_name)
    join = args.separator.join
    prefix = f"{'int64' if args.int64 else 'bytes'}_{args.name}_"
    encode = (GPT2TokenizerFast.from_pretrained('gpt2') if args.int64 else str).encode

    files_processed = 0
    tfrecord_count = 0
    chunk = 0
    buffer_size = 0
    tokenized_files = []

    last_write = start_time = time.time()

    for f in file_generator(args, pid, procs):
        buffer_size += len(f)
        tokenized_files.append(f)
        files_processed += 1

        if buffer_size > chunk * args.buffer_size // 4:
            print(f"Worker: {pid:{len(str(procs))}d} | Buffer: {buffer_size * 2 ** -20:.1f}MB | "
                  f"Files: {files_processed} - TFrecords: {tfrecord_count} | "
                  f"Wrote: {time.time() - last_write:.0f}s ago - Started: {time.time() - start_time:.0f}s ago",
                  end='')
            chunk += 1

        if buffer_size > args.buffer_size:
            filename = f"{prefix}{tfrecord_count:_>6d}_{files_processed}_{buffer_size}.tfrecord"

            joined = encode(join(tokenized_files))
            tokenized_files.clear()

            with tf.io.TFRecordWriter(filename) as writer:
                if args.int64:
                    feature = {"text": tf.train.Feature(int64_list=tf.train.Int64List(value=joined))}
                else:
                    feature = {"text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[joined]))}
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())

            bucket.blob(f'{output_dir}{filename}').upload_from_filename(filename)

            os.remove(filename)
            chunk = 0
            buffer_size = 0
            tfrecord_count += 1

            print("")

            last_write = time.time()


def main():
    args = parser.parse_args()

    if not args.output_dir.endswith("/"):
        args.output_dir = args.output_dir + "/"
    if not args.output_dir.startswith("gs://"):
        print("Output dir isn't a cloud bucket. Exiting.")
        return
    args.output_dir = args.output_dir[len('gs://'):]
    processes = [multiprocessing.Process(target=create_tfrecords, args=(args, pid, args.procs)) for pid in
                 range(args.procs)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
