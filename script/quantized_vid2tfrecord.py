import argparse
import datetime
import functools
import io
import multiprocessing
import os
import pickle
import queue
import random
import subprocess
import sys
import threading
import time
import typing
from multiprocessing.shared_memory import SharedMemory

import boto3
import cv2
import gdown
import numpy as np
import requests
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
    parser.add_argument("--startup-delay", type=int, default=10,
                        help="Seconds to wait after launching one worker (to avoid crashes)")
    parser.add_argument("--prefetch", type=int, default=8, help="Number of videos to prefetch (default=8)")
    args = parser.parse_args()
    return args.cpu_worker, args.bucket, args.prefix, args.tmp_dir, args.urls, args.fps, args.startup_delay, \
           args.batch, args.device, args.prefetch, args.model_base_path, args.shared_memory


def division_zero(x, y):
    return 0 if y == 0 else (x / y)


class Downloader:
    def __init__(self, max_try: int = 3):
        self.max_try = max_try

    def download(self, url: str, filename: str):
        for _ in range(self.max_try):
            try:
                r = requests.get(url, stream=True)
                with open(filename, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
            except:
                pass
            else:
                return True

        print(f'Retry exceeded for URL: {url}')

        if os.path.exists(filename):
            os.remove(filename)

        return False


def frame_encoder(frame):
    feature = {'text': tf.train.Feature(int64_list=tf.train.Int64List(value=frame))}
    features = tf.train.Features(feature=feature)
    proto = tf.train.Example(features=features)
    proto = proto.SerializeToString()
    return proto


def try_except(fn, default=None):
    def _fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(e)
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
    if 'formats' not in info:
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


def test_video(video_buffer_path: str):
    video_cap = None
    try:
        video_cap = cv2.VideoCapture(video_buffer_path)
        success, frame = video_cap.read()
        video_cap.release()
    except:
        success = False
    if video_cap is not None:
        video_cap.release()
    return success


@try_except
def download_video(video_urls: typing.List[dict], downloader: Downloader, worker_id: int, download_buffer_dir: str,
                   yt_url: str) -> str:
    # Put .webm at the bottom at the list.
    for idx in range(len(video_urls)):
        if video_urls[idx]['ext'] == 'webm':
            video_urls[-1], video_urls[idx] = video_urls[idx], video_urls[-1]

    for video_url_idx, video_url in enumerate(video_urls):
        url = video_url['url']
        ext = video_url['ext']

        if url is None or ext is None or url == "" or ext == "":
            continue

        video_buffer_path = os.path.join(download_buffer_dir, yt_url) + '.' + ext
        if not downloader.download(url, video_buffer_path):
            continue
        # If no mp4 got downloaded use ffmpeg to converted it to mp4
        if ext != 'mp4':
            new_video_buffer_path = os.path.join(download_buffer_dir, yt_url) + '.mp4'
            subprocess.run(['ffmpeg', '-i', video_buffer_path, '-c', 'copy', new_video_buffer_path, '-y'],
                           capture_output=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            if os.path.exists(video_buffer_path):
                os.remove(video_buffer_path)

            video_buffer_path = new_video_buffer_path

        # Check if the file can be opened.
        if not test_video(video_buffer_path) and os.path.exists(video_buffer_path):
            os.remove(video_buffer_path)
        else:
            return video_buffer_path
    raise ValueError("worker: " + str(worker_id) + " failed to download video")


@try_except
def get_video_frames(path: str, target_image_size: int, target_fps: int):
    frames = []
    frame_idx = 0
    success = True
    video_cap = cv2.VideoCapture(path)
    fps_split = division_zero(round(video_cap.get(cv2.CAP_PROP_FPS)), target_fps)
    while success:
        success, frame = video_cap.read()
        if frame_idx % fps_split == 0:
            frames.append(cv2.resize(frame, (target_image_size, target_image_size)))
        frame_idx += 1
    video_cap.release()
    return frames


@functools.partial(try_except, default=0)
def write_numpy(tokens: typing.List[int], buffer_save_dir: str, save_dir: str, s3_bucket):
    with io.BytesIO() as f:
        np.save(f, np.array(tokens))
        s3_bucket.put_object(Body=f, Key=f"{buffer_save_dir}/{save_dir.replace('/', '_')}.npy")
    tokens.clear()
    return len(tokens)


def log_fn(*args, worker_id: int):
    print(f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']} - worker:{worker_id:2d} - {datetime.datetime.now()}", *args)


def frame_worker(work: list, worker_id: int, lock: threading.Lock, target_image_size: int,
                 download_buffer_dir: str, target_fps: int, batch_size: int,
                 index_mem_name: str, frame_mem_name: str, read_shared_lock: threading.Lock,
                 write_shared_lock: threading.Lock, shape: typing.Tuple[int]):

    log = functools.partial(log_fn, worker_id=worker_id)
    log("starting frame worker")

    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL(
            {'writeautomaticsub': False, 'ignore-errors': True, 'socket-timeout': 600, "quiet": True, "verbose": False,
             "no_warnings": True
             })
    youtube_getter.add_default_info_extractors()
    downloader = Downloader()
    random.Random(worker_id).shuffle(work)

    shared_index_mem = SharedMemory(create=False, name=index_mem_name)
    shared_frame_mem = SharedMemory(create=False, name=frame_mem_name)
    index_mem = np.ndarray((256, 2), dtype=np.uint32, buffer=shared_index_mem.buf)
    frame_mem = np.ndarray(shape, dtype=np.uint8, buffer=shared_frame_mem.buf)

    for wor in work:
        log(wor)
        video_urls = get_video_urls(youtube_getter, youtube_base, wor, lock, target_image_size)
        if not video_urls:
            log("no urls")
            continue

        path = download_video(video_urls, downloader, worker_id, download_buffer_dir, wor)
        if not path or not test_video(path):
            log("no path")
            continue

        frames = get_video_frames(path, target_image_size, target_fps)
        if not frames:
            log("no frames")
            continue
        os.remove(path)

        cv2_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        frames: np.ndarray = np.stack(cv2_frames).astype(np.uint8).transpose((0, 3, 1, 2))
        frames = frames[:frames.shape[0] // batch_size * batch_size]
        if not frames.size:
            log("no frames")
            continue
        frames = frames.reshape((-1, batch_size, 3, target_image_size, target_image_size))
        log("put")
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

        log("in the queue")


def worker(model: GumbelVQ, save_dir: str, download_buffer_dir: str, bucket_name: str, device: torch.device,
           index: np.ndarray, shared_frames: np.ndarray, read_shared_lock: threading.Lock):
    print(os.environ["CUDA_VISIBLE_DEVICES"], "starting worker")
    torch.set_default_tensor_type('torch.FloatTensor')
    s3_bucket = boto3.resource("s3").Bucket(bucket_name)
    model = model.to(device)
    total_frames = 0
    waiting = 0
    tokens = []
    log = functools.partial(log_fn, worker_id=-1)
    while waiting < 120:
        log(f"Tokens: {len(tokens):,d} - Frames: {total_frames:,d}")
        while index[:, 1].max() == 0 and waiting < 120:  # wait until one element exists
            time.sleep(5)
            waiting += 1
        if waiting >= 120:
            log("done. dumping now")
            break
        with read_shared_lock:  # lock reader, so it won't move memory while we're copying
            idx = index[:, 1].argmax()  # pick first
            start, end = index[idx]
            frames = shared_frames[start:end].copy()  # local clone, so it shared can be safely edited
            index[idx] = [-1, 0]  # reset indices (-1 -> 2^32-1, so it won't map to "min" in frame_worker)
        frames = torch.as_tensor(frames.astype(np.float32) / 255)
        total_frames += frames.size(0) * frames.size(1)
        tokens.extend(tokenize(model, frames, device))
        waiting = 0
    write_numpy(tokens, download_buffer_dir, save_dir, s3_bucket)


def main():
    workers, bucket, prefix, tmp_dir, urls, fps, startup_delay, batch_size, device, prefetch, model_path, \
    shared_memory = parse_args()
    config_path = f'{model_path}/vqgan.gumbelf8.config.yml'
    model_path = f'{model_path}/sber.gumbelf8.ckpt'
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1WP6Li2Po8xYcQPGMpmaxIlI1yPB5lF5m', model_path, quiet=True)
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1M7RvSoiuKBwpF-98sScKng0lsZnwFebR', config_path, quiet=True)
    os.makedirs(tmp_dir, exist_ok=True)
    conf = OmegaConf.load(config_path)
    resolution = conf.model.params.ddconfig.resolution
    model = load_vqgan(config_path, model_path)

    shared_memory = shared_memory * 1024 ** 3  # it's in GB, we have to convert it to bytes
    shared_frames = shared_memory // (256 ** 2 * 3 * batch_size)
    index = np.zeros((256, 2), dtype=np.uint32)  # 256x start+end
    shape = (shared_frames, batch_size, 3, 256, 256)
    frames = np.zeros(shape, dtype=np.uint8)
    index_mem = SharedMemory(create=True, size=index.nbytes)
    frame_mem = SharedMemory(create=True, size=frames.nbytes)
    index = np.ndarray((256, 2), dtype=np.uint32, buffer=index_mem.buf)
    frame = np.ndarray(shape, dtype=np.uint8, buffer=frame_mem.buf)
    index[:] = 0
    frame[:] = 0

    with open(urls, 'rb') as f:
        video_ids, _ = pickle.load(f)

    ids = [video_ids[int(len(video_ids) * i / workers):int(len(video_ids) * (i + 1) / workers)] for i in range(workers)]

    lock = multiprocessing.Lock()
    read_shared_lock = multiprocessing.Lock()
    write_shared_lock = multiprocessing.Lock()

    procs = [multiprocessing.Process(args=(
            work, worker_id, lock, resolution, tmp_dir, fps, batch_size, index_mem.name, frame_mem.name,
            read_shared_lock, write_shared_lock, shape),
            daemon=True, target=frame_worker) for worker_id, work in enumerate(ids)]
    for p in procs:
        p.start()

    return worker(model, prefix, tmp_dir, bucket, torch.device(device), index, frame, read_shared_lock)


if __name__ == '__main__':
    main()
