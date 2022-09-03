import argparse
import copy
import datetime
import functools
import multiprocessing
import os
import pickle
import random
import shutil
import sys
import threading
import time
import traceback
import typing
import uuid

import boto3
import ffmpeg
import gdown
import numpy as np
import requests
import tensorflow as tf
import torch
import youtube_dl
from omegaconf import OmegaConf
from sharedutils import SharedEXTQueue

sys.path.append("./taming-transformers")
from taming.models.vqgan import GumbelVQ  # skipcq: FLK-E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-worker", type=int, default=multiprocessing.cpu_count(),
                        help=f"Number of workers. Default is the number of CPU cores (={multiprocessing.cpu_count()})")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model-base-path", type=str, default='/fsx/lucas',
                        help="Where model and config should be dowloaded to")
    parser.add_argument("--bucket", type=str, help="Name of the S3 bucket")
    parser.add_argument("--prefix", type=str, help="Prefix in the bucket")
    parser.add_argument("--batch", type=int, default=128, help="Number of images processed per 'computation step'")
    parser.add_argument("--tmp-dir", type=str, help="Local directory for temporary storage")
    parser.add_argument("--urls", type=str, help="Directory filled with JSON files full of URLs")
    parser.add_argument("--fps", type=int, default=1,
                        help="Number of (encoded) video frames per second of raw data (default=4)")
    parser.add_argument("--shared-memory", type=int, default=4, help="number of GB of shared memory")
    parser.add_argument("--tokens-per-file", type=int, default=2 ** 28, help="how big each file should roughly be")
    parser.add_argument("--video-downloaders", type=int, default=4,
                        help="Number of parallel video _information_ downloaders. Videos are always downloaded in "
                             "parallel, but downloading information about too many videos in parallel can lead to "
                             "errors and slow things down.")
    args = parser.parse_args()
    return args.cpu_worker, args.bucket, args.prefix, args.tmp_dir, args.urls, args.fps, args.batch, args.gpus, \
           args.model_base_path, args.shared_memory, args.tokens_per_file, args.video_downloaders


def frame_encoder(frame):
    feature = {'text': tf.train.Feature(int64_list=tf.train.Int64List(value=frame))}
    features = tf.train.Features(feature=feature)
    proto = tf.train.Example(features=features)
    proto = proto.SerializeToString()
    return proto


def try_except(fn: typing.Callable, default=None):
    def _fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # skipcq: PYL-W0703
            print(r"IGNORED EXCEPTION \/\/\/")
            print(fn, exc)
            traceback.print_exc()
            print("IGNORED EXCEPTION /\\/\\/\\")

        return default

    return _fn


def load_vqgan(config_path: str, ckpt_path: str):
    config = OmegaConf.load(config_path)
    model = GumbelVQ(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval()


@functools.partial(try_except, default=[])
def tokenize(model: GumbelVQ, frames: torch.Tensor, device: torch.device):
    with torch.no_grad():
        batches = [model.encode(f.to(device))[2][2].detach() for f in frames]
        return torch.cat(batches, dim=0).flatten().cpu().tolist()


@try_except
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: threading.Semaphore, target_image_size: int) -> \
        typing.List[dict]:
    # We have to lock this part because it can lead to errors if multiple thread try to scrape video Information at
    # the same time.
    with lock:
        info = youtube_getter.extract_info(youtube_base + url, download=False)
    if info is None or 'formats' not in info:
        return []
    video_urls = []
    for f in info['formats']:
        width = f.get('width')
        height = f.get('height')
        url = f.get('url')
        ext = f.get('ext')
        format_note = f.get('format_note')

        if any(x is None for x in (width, height, url, ext, format_note)):
            continue
        if any(not x for x in (width, height, url, ext)):
            continue
        if format_note == "tiny" or width <= target_image_size or height <= target_image_size:
            continue
        video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url']})
    return sorted(video_urls, key=lambda x: (x['ext'] != 'mp4', x['width'], x['height']))


def get_video_frames(video_urls: typing.List[dict], target_image_size: int, target_fps: int) -> np.ndarray:
    filename = uuid.uuid4()
    path = str(filename)
    for video_url in video_urls:
        if os.path.exists(path):
            os.remove(path)

        url = video_url["url"]
        path = f"{filename}.{video_url['ext']}"

        try:
            with requests.get(url, stream=True) as r, open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        except Exception:  # skipcq: PYL-W0703
            continue  # Broken URL, next might work

        width = round(video_url["width"] * video_url["height"] / target_image_size)
        try:
            out, _ = ffmpeg.input(path) \
                .filter("scale", w=width, h=target_image_size) \
                .filter("crop", w=target_image_size, h=target_image_size).filter("fps", target_fps) \
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error", preset="ultrafast",
                        threads=target_image_size // 40) \
                .run(capture_stdout=True)
        except ffmpeg.Error:  # Broken Video, next might work
            continue

        if os.path.exists(path):
            os.remove(path)
        return np.frombuffer(out, np.uint8).reshape((-1, target_image_size, target_image_size, 3))


@functools.partial(try_except, default=0)
def write_tfrecords(tokens: typing.List[int], chunk_size: int, buffer_save_dir: str, save_dir: str, tfrecord_id: int,
                    s3_bucket):
    path = f"{buffer_save_dir}/{save_dir.replace('/', '_')}_{tfrecord_id}.tfrecord"
    count = len(tokens)
    residual = count % chunk_size
    count -= residual
    if not count:
        return 0

    added = 0

    for i in range(0, count, chunk_size):
        with tf.io.TFRecordWriter(path) as tf_writer:
            tf_writer.write(frame_encoder(tokens[i:i + chunk_size]))
        s3_bucket.upload_file(path, f"{save_dir.rstrip('/')}/{tfrecord_id + added:07d}.tfrecord")
        os.remove(path)
        added += 1
    residual_tokens = tokens[-residual:]
    tokens.clear()
    tokens.extend(residual_tokens)
    return added


def frame_worker(work: list, worker_id: int, lock: threading.Semaphore, target_image_size: int, target_fps: int,
                 batch_size: int, queue_export):
    queue = SharedEXTQueue.from_export(*queue_export)
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
            {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
             "ignoreerrors": True
             })
    youtube_getter.add_default_info_extractors()
    random.Random(worker_id).shuffle(work)

    for wor in work:
        video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size)

        if not video_urls:
            continue

        frames = get_video_frames(video_urls, target_image_size, target_fps)

        if frames is None or not frames.size:
            continue

        frames: np.ndarray = frames
        frames = frames[:frames.shape[0] // batch_size * batch_size]
        frames = frames.transpose((0, 3, 1, 2)).reshape((-1, batch_size, 3, target_image_size, target_image_size))
        queue.put(frames)


def worker(model: GumbelVQ, save_dir: str, download_buffer_dir: str, bucket, device: int,
           queue: SharedEXTQueue, procs: typing.List[multiprocessing.Process], tokens_per_file: int,
           padding_token: int):
    save_dir = f'{save_dir.rstrip("/")}/{device}'
    dev_str = f'cuda:{device}'
    device = torch.device(dev_str)
    torch.set_default_tensor_type('torch.FloatTensor')
    model = copy.deepcopy(model)
    model = model.to(device)
    total_frames = 0
    tokens = []
    tfrecord_id = 0
    start_time = time.time()
    start = datetime.datetime.now()
    token_pad = len(f'{tokens_per_file:,d}')
    frame_pad = len(f'{tokens_per_file // 1024:,d}')
    while True:
        print(f"{dev_str} | {datetime.datetime.now()} | Tokens: {len(tokens):{token_pad},d} - "
              f"Frames: {total_frames:{frame_pad},d} | "
              f"FramesPerSecond: {total_frames / (time.time() - start_time):5.2f} - "
              f"Elapsed: {datetime.datetime.now() - start}", flush=True)

        # wait until one element exists or run is over
        while not queue and any(p.is_alive() for p in procs):
            time.sleep(1)
        if not any(p.is_alive() for p in procs):
            break
        frames = queue.get()
        frames = torch.as_tensor(frames.astype(np.float32) / 255)
        total_frames += frames.size(0) * frames.size(1)
        if tokens:
            tokens.append(padding_token)
        tokens.extend(tokenize(model, frames, device))
        tfrecord_id += write_tfrecords(tokens, tokens_per_file, download_buffer_dir, save_dir, tfrecord_id, bucket)
    write_tfrecords(tokens, tokens_per_file, download_buffer_dir, save_dir, tfrecord_id, bucket)


def main():
    workers, bucket, prefix, tmp_dir, urls, fps, batch_size, gpus, model_path, shared_memory, chunk_size, \
    video_downloaders = parse_args()
    config_path = f'{model_path}/vqgan.gumbelf8.config.yml'
    model_path = f'{model_path}/sber.gumbelf8.ckpt'
    if not os.path.exists(config_path):
        gdown.download('https://drive.google.com/uc?id=1WP6Li2Po8xYcQPGMpmaxIlI1yPB5lF5m', model_path, quiet=True)
    if not os.path.exists(config_path):
        gdown.download('https://drive.google.com/uc?id=1M7RvSoiuKBwpF-98sScKng0lsZnwFebR', config_path, quiet=True)
    os.makedirs(tmp_dir, exist_ok=True)
    conf = OmegaConf.load(config_path)
    padding_token = conf.model.params.n_embed
    resolution = conf.model.params.ddconfig.resolution
    model = load_vqgan(config_path, model_path)

    shared_memory = shared_memory * 1024 ** 3  # it's in GB, we have to convert it to bytes
    shared_frames = shared_memory // (256 ** 2 * 3 * batch_size)
    queue = SharedEXTQueue.from_shape([shared_frames, batch_size, 3, 256, 256])

    ids = []
    for path in os.listdir(urls):
        with open(f'{urls}/{path}', 'rb') as f:
            video_ids, _ = pickle.load(f)  # skipcq: BAN-B301
            ids.extend(video_ids)

    ids = [ids[int(len(ids) * i / workers):int(len(ids) * (i + 1) / workers)] for i in range(workers)]
    lock = multiprocessing.Semaphore(video_downloaders)
    procs = [multiprocessing.Process(args=(work, worker_id, lock, resolution, fps, batch_size, queue.export()),
                                     daemon=True, target=frame_worker) for worker_id, work in enumerate(ids)]
    for p in procs:
        p.start()

    while not queue:  # "pre-wait" to get more accurate FPS counters
        time.sleep(1)

    bucket = boto3.resource("s3").Bucket(bucket)
    threads = [threading.Thread(target=worker,
                                args=(model, prefix, tmp_dir, bucket, i, queue, procs, chunk_size, padding_token),
                                daemon=True)
               for i in range(gpus)]

    for t in threads:
        t.start()

    for p in procs + threads:
        p.join()

    queue.frame_mem.unlink()
    queue.frame_mem.close()


if __name__ == '__main__':
    main()
