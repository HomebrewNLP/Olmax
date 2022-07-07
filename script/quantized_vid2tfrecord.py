import argparse
import datetime
import functools
import multiprocessing
import os
import pickle
import random
import sys
import threading
import time
import traceback
import typing
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing.shared_memory import SharedMemory

import boto3
import ffmpeg
import gdown
import numpy as np
import tensorflow as tf
import torch
import youtube_dl
from omegaconf import OmegaConf

sys.path.append("./taming-transformers")
from taming.models.vqgan import GumbelVQ


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-worker", type=int, default=multiprocessing.cpu_count(),
                        help=f"Number of workers. Default is the number of CPU cores (={multiprocessing.cpu_count()})")
    parser.add_argument("--device", type=str, default='cuda:0', help="Whether to use GPU or CPU. Default is GPU.")
    parser.add_argument("--model-base-path", type=str, default='/fsx/lucas',
                        help="Where model and config should be dowloaded to")
    parser.add_argument("--bucket", type=str, help="Name of the S3 bucket")
    parser.add_argument("--prefix", type=str, help="Prefix in the bucket")
    parser.add_argument("--batch", type=int, default=64, help="Number of images processed per 'computation step'")
    parser.add_argument("--tmp-dir", type=str, help="Local directory for temporary storage")
    parser.add_argument("--urls", type=str, help="Directory filled with JSON files full of URLs")
    parser.add_argument("--fps", type=int, default=1,
                        help="Number of (encoded) video frames per second of raw data (default=4)")
    parser.add_argument("--shared-memory", type=int, default=4, help="number of GB of shared memory")
    parser.add_argument("--tokens-per-file", type=int, default=2 ** 29, help="how big each file should roughly be")
    parser.add_argument("--startup-delay", type=int, default=10,
                        help="Seconds to wait after launching one worker (to avoid crashes)")
    args = parser.parse_args()
    return args.cpu_worker, args.bucket, args.prefix, args.tmp_dir, args.urls, args.fps, args.startup_delay, \
           args.batch, args.device, args.model_base_path, args.shared_memory, args.tokens_per_file


def frame_encoder(frame):
    feature = {'text': tf.train.Feature(int64_list=tf.train.Int64List(value=frame))}
    features = tf.train.Features(feature=feature)
    proto = tf.train.Example(features=features)
    proto = proto.SerializeToString()
    return proto


def try_except(fn: typing.Callable, default=None):
    def _fn(*args, **kwargs):
        try:
            with open(os.devnull, 'w') as fnull, redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                return fn(*args, **kwargs)
        except Exception:
            print(r"IGNORED EXCEPTION \/\/\/")
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
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: multiprocessing.Lock, target_image_size: int) -> \
        typing.List[dict]:
    # We have to lock this part because it can lead to errors if multiple thread try to
    # scrap video Information at the same time.
    with lock:
        info = youtube_getter.extract_info(youtube_base + url, download=False)
    if info is None or 'formats' not in info:
        return []
    video_urls = []
    current_width = current_height = 9999999
    for f in info['formats']:
        if 'format_note' not in f or f['format_note'] == "tiny" or 'width' not in f or 'height' not in f:
            continue
        width = f['width']
        height = f['height']

        if width is None or height is None or width <= target_image_size or height <= target_image_size:
            continue
        if current_width > width and current_height > height:
            video_urls = []
            current_width = width
            current_height = height
        if current_width == width and current_height == height:
            if 'ext' in f and 'url' in f:
                video_urls.append({'width': width, 'height': height, 'ext': f['ext'], 'url': f['url']})
    return video_urls


@functools.partial(try_except, default=[])
def get_video_frames(video_urls: typing.List[dict], target_image_size: int, target_fps: int):
    # Put .webm at the bottom at the list.
    for idx in range(len(video_urls)):
        if video_urls[idx]['ext'] == 'webm':
            video_urls[-1], video_urls[idx] = video_urls[idx], video_urls[-1]

    for video_url_idx, video_url in enumerate(video_urls):
        url = video_url.get('url', None)
        if url is None or url == "":
            continue
        out, _ = ffmpeg.input(url).filter("scale", w=-1, h=target_image_size) \
            .filter("crop", w=target_image_size, h=target_image_size).filter("fps", target_fps) \
            .output("pipe:", format="rawvideo", pix_fmt="rgb48", loglevel="error").run(capture_stdout=True)
        return np.frombuffer(out, np.uint16).reshape((-1, target_image_size, target_image_size, 3))


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


def log_fn(*args, worker_id: int):
    print(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']} - worker:{worker_id:2d} - {datetime.datetime.now()}", *args,
          flush=True)


def frame_worker(work: list, worker_id: int, lock: threading.Lock, target_image_size: int, target_fps: int,
                 batch_size: int, index_mem_name: str, frame_mem_name: str, read_shared_lock: threading.Lock,
                 write_shared_lock: threading.Lock, shape: typing.Tuple[int]):

    log = functools.partial(log_fn, worker_id=worker_id)
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
            {'writeautomaticsub': False, 'socket_timeout': 600, "quiet": True, "verbose": False, "no_warnings": True,
             "ignoreerrors": True
             })
    youtube_getter.add_default_info_extractors()
    random.Random(worker_id).shuffle(work)

    shared_index_mem = SharedMemory(create=False, name=index_mem_name)
    shared_frame_mem = SharedMemory(create=False, name=frame_mem_name)
    index_mem = np.ndarray((256, 2), dtype=np.uint32, buffer=shared_index_mem.buf)
    frame_mem = np.ndarray(shape, dtype=np.uint16, buffer=shared_frame_mem.buf)

    for wor in work:
        video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size)

        if not video_urls:
            continue

        frames = get_video_frames(video_urls, target_image_size, target_fps)
        if not frames:
            continue

        frames: np.ndarray = frames
        frames = frames[:frames.shape[0] // batch_size * batch_size]
        if not frames.size:
            continue
        frames = frames.transpose((0, 3, 1, 2)).reshape((-1, batch_size, 3, target_image_size, target_image_size))
        batch_count = frames.shape[0]
        if batch_count >= frame_mem.shape[0]:
            log(f"dropping {wor}. too many batches in video (={batch_count}) compared to max memory size "
                f"(={frame_mem.shape[0]})")

        with write_shared_lock:
            while index_mem[:, 1].max() + batch_count >= frame_mem.shape[0]:  # until new frames fit into memory
                while index_mem[:, 0].min() == 0:  # wait for anything to be read
                    time.sleep(5)
                with read_shared_lock:  # move everything to the left  (old=left, new=right)
                    min_start = index_mem[:, 0].min()
                    max_end = index_mem[:, 1].max()
                    dist = max_end - min_start
                    frame_mem[:dist] = frame_mem[min_start:max_end]
                    start_idx = index_mem[:, 0].argmin()
                    end_idx = index_mem[:, 1].argmax()
                    index_mem[:end_idx - start_idx] = index_mem[start_idx:end_idx] - min_start
                    index_mem[end_idx - start_idx:] = 0
            max_end = index_mem[:, 1].max()
            end_idx = index_mem[:, 1].argmax()
            frame_mem[max_end:max_end + batch_count] = frames[:]
            if max_end != 0:  # if array is empty, make sure to use the first (0th) spot
                end_idx += 1
            index_mem[end_idx] = [max_end, max_end + batch_count]


def worker(model: GumbelVQ, save_dir: str, download_buffer_dir: str, bucket_name: str, device: torch.device,
           index: np.ndarray, shared_frames: np.ndarray, read_shared_lock: threading.Lock,
           procs: typing.List[multiprocessing.Process], tokens_per_file: int, padding_token: int):
    print(os.environ["CUDA_VISIBLE_DEVICES"], "starting worker")
    torch.set_default_tensor_type('torch.FloatTensor')
    s3_bucket = boto3.resource("s3").Bucket(bucket_name)
    model = model.to(device)
    total_frames = 0
    waiting = 0
    tokens = []
    log = functools.partial(log_fn, worker_id=-1)
    tfrecord_id = 0
    while True:
        log(f"Tokens: {len(tokens):,d} - Frames: {total_frames:,d}")
        # wait until one element exists or run is over
        while index[:, 1].max() == 0 and any(p.is_alive() for p in procs):
            time.sleep(5)
            waiting += 1
        if not any(p.is_alive() for p in procs):
            log("Finished")
            break
        with read_shared_lock:  # lock reader, so it won't move memory while we're copying
            idx = index[:, 1].argmax()  # pick first
            start, end = index[idx]
            frames = shared_frames[start:end].copy()  # local clone, so it shared can be safely edited
            index[idx] = [-1, 0]  # reset indices (-1 -> 2^32-1, so it won't map to "min" in frame_worker)
        frames = torch.as_tensor(frames.astype(np.float32) / 65535)
        total_frames += frames.size(0) * frames.size(1)
        if tokens:
            tokens.append(padding_token)
        tokens.extend(tokenize(model, frames, device))
        waiting = 0
        tfrecord_id += write_tfrecords(tokens, tokens_per_file, download_buffer_dir, save_dir, tfrecord_id, s3_bucket)
    write_tfrecords(tokens, tokens_per_file, download_buffer_dir, save_dir, tfrecord_id, s3_bucket)


def main():
    workers, bucket, prefix, tmp_dir, urls, fps, startup_delay, batch_size, device, model_path, \
    shared_memory, tokens_per_file = parse_args()
    config_path = f'{model_path}/vqgan.gumbelf8.config.yml'
    model_path = f'{model_path}/sber.gumbelf8.ckpt'
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1WP6Li2Po8xYcQPGMpmaxIlI1yPB5lF5m', model_path, quiet=True)
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1M7RvSoiuKBwpF-98sScKng0lsZnwFebR', config_path, quiet=True)
    os.makedirs(tmp_dir, exist_ok=True)
    conf = OmegaConf.load(config_path)
    padding_token = conf.model.params.n_embed
    resolution = conf.model.params.ddconfig.resolution
    model = load_vqgan(config_path, model_path)

    shared_memory = shared_memory * 1024 ** 3  # it's in GB, we have to convert it to bytes
    shared_frames = shared_memory // (256 ** 2 * 3 * batch_size)
    index = np.zeros((256, 2), dtype=np.uint32)  # 256x start+end
    shape = (shared_frames, batch_size, 3, 256, 256)
    frames = np.zeros(shape, dtype=np.uint16)
    index_mem = SharedMemory(create=True, size=index.nbytes)
    frame_mem = SharedMemory(create=True, size=frames.nbytes)
    index = np.ndarray((256, 2), dtype=np.uint32, buffer=index_mem.buf)
    frame = np.ndarray(shape, dtype=np.uint16, buffer=frame_mem.buf)
    index[:] = 0
    frame[:] = 0

    with open(urls, 'rb') as f:
        video_ids, _ = pickle.load(f)

    ids = [video_ids[int(len(video_ids) * i / workers):int(len(video_ids) * (i + 1) / workers)] for i in range(workers)]

    lock = multiprocessing.Lock()
    read_shared_lock = multiprocessing.Lock()
    write_shared_lock = multiprocessing.Lock()

    procs = [multiprocessing.Process(args=(
            work, worker_id, lock, resolution, fps, batch_size, index_mem.name, frame_mem.name,
            read_shared_lock, write_shared_lock, shape),
            daemon=True, target=frame_worker) for worker_id, work in enumerate(ids)]
    for p in procs:
        p.start()

    return worker(model, prefix, tmp_dir, bucket, torch.device(device), index, frame, read_shared_lock, procs,
                  tokens_per_file, padding_token)


if __name__ == '__main__':
    main()
