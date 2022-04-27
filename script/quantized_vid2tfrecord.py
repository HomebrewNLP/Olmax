import argparse
import datetime
import functools
import json
import multiprocessing
import os
import queue
import random
import subprocess
import sys
import threading
import typing

import cv2
import gdown
import numpy as np
import requests
import tensorflow as tf
import torch
import youtube_dl
from google.cloud import storage
from omegaconf import OmegaConf

sys.path.append("./taming-transformers")
from taming.models.vqgan import GumbelVQ


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-worker",
        type=int,
        default=multiprocessing.cpu_count(),
        help=f"Number of workers. Default is the number of CPU cores (={multiprocessing.cpu_count()})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Whether to use GPU or CPU. Default is GPU."
    )
    parser.add_argument(
        "--service-account-json",
        type=str,
        default='',
        help="Path to service account json file. default=use service acccount of machine"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="Name of the GCS bucket"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix in the bucket"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=48,
        help="Number of images processed per 'computation step'"
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        help="Local directory for temporary storage"
    )
    parser.add_argument(
        "--urls",
        type=str,
        help="Directory filled with JSON files full of URLs"
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=60 * 4,
        help="Minimum video duration in seconds (default=4 minutes)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512 * 1024 * 1024,
        help="Number of tokens per tfrecord (default=512 million tokens)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Number of (encoded) video frames per second of raw data (default=4)"
    )
    parser.add_argument(
        "--startup-delay",
        type=int,
        default=10,
        help="Seconds to wait after launching one worker (to avoid crashes)"
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="Number of videos to prefetch (default=8)"
    )
    args = parser.parse_args()
    return args.cpu_worker, args.bucket, args.prefix, args.tmp_dir, args.urls, args.min_duration, args.chunk_size, args.fps, args.startup_delay, args.batch, args.service_account_json, args.device, args.prefetch


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


def split_equal(ids: list, duration: list, num: int, min_duration: int = 256):
    sort = sorted(zip(duration, ids))[::-1]

    ids_split = [[] for _ in range(num)]
    duration_spit = [[] for _ in range(num)]
    duration_sum = [0] * num

    for d, i in sort:
        if d > min_duration or min_duration <= 0:
            pos = np.argmin(duration_sum)

            ids_split[pos].append(i)
            duration_spit[pos].append(d)
            duration_sum[pos] = duration_sum[pos] + d

    return ids_split, duration_spit


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
def get_video_urls(youtube_getter, youtube_base: str, url: str, lock: multiprocessing.Lock,
                   target_image_size: int) -> typing.List[dict]:
    print(youtube_base + url)
    # We have to lock this part because it can lead to errors if multiple thread try to
    # scrap video Information at the same time.
    with lock:
        info = youtube_getter.extract_info(youtube_base + url, download=False)
    if 'formats' not in info:
        return []
    video_urls = []
    current_width = current_height = 9999999
    for f in info['formats']:
        if 'format_note' not in f or f['format_note'] == "tiny" or 'width' not in f or \
                'height' not in f:
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
            subprocess.run(['ffmpeg', '-i', video_buffer_path, '-c',
                            'copy', new_video_buffer_path, '-y'],
                           capture_output=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

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
def write_tfrecords(tokens: typing.List[int], chunk_size: int, buffer_save_dir: str, save_dir: str, tfrecord_id: int,
                    padding_token: int, cloud_storage_bucket):
    path = f"{buffer_save_dir}/{save_dir.replace('/', '_')}_{tfrecord_id}.tfrecord"
    count = len(tokens)
    residual = count % chunk_size
    count -= residual
    added = 0

    for i in range(0, count, chunk_size):
        with tf.io.TFRecordWriter(path) as tf_writer:
            tf_writer.write(frame_encoder(tokens[i:i + chunk_size]))
        blob = cloud_storage_bucket.blob(f'{save_dir}{tfrecord_id + added:07d}.tfrecord')
        blob.upload_from_filename(path)
        os.remove(path)
        added += 1
    residual_tokens = tokens[-residual:] + [padding_token]
    tokens.clear()
    tokens.extend(residual_tokens)
    return added


def frame_worker(work: list, worker_id: int, lock: threading.Lock, target_image_size: int, download_buffer_dir: str,
                 target_fps: int, batch_size: int, out_queue: queue.Queue):
    youtube_base = 'https://www.youtube.com/watch?v='
    youtube_getter = youtube_dl.YoutubeDL({'writeautomaticsub': False, 'ignore-errors': True, 'socket-timeout': 600})
    youtube_getter.add_default_info_extractors()
    downloader = Downloader()
    random.Random(worker_id).shuffle(work)

    for chunk_idx, wor in enumerate(work):
        for wor_idx, _wor in enumerate(wor):
            video_urls = get_video_urls(youtube_getter, youtube_base, _wor, lock, target_image_size)
            if not video_urls:
                continue

            path = download_video(video_urls, downloader, worker_id, download_buffer_dir, _wor)
            if not path or not test_video(path):
                continue

            frames = get_video_frames(path, target_image_size, target_fps)
            if not frames:
                continue
            os.remove(path)

            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            frames = np.stack(frames).astype(np.float32).transpose((0, 3, 1, 2)) / 255
            frames = frames[:frames.shape[0] // batch_size * batch_size]
            frames = frames.reshape((-1, batch_size, 3, target_image_size, target_image_size))
            frames = torch.from_numpy(frames)

            out_queue.put(frames)


def worker(model: GumbelVQ,
           chunk_size: int,
           save_dir: str,
           download_buffer_dir: str,
           bucket_name: str,
           padding_token: int,
           service_account_json: str,
           device: torch.device,
           frame_queue: queue.Queue):
    torch.set_default_tensor_type('torch.FloatTensor')

    if service_account_json:
        cloud_storage_bucket = storage.Client.from_service_account_json(service_account_json).get_bucket(bucket_name)
    else:
        cloud_storage_bucket = storage.Client().get_bucket(bucket_name)

    model = model.to(device)

    tfrecord_id = 0
    total_frames = 0
    tokens = []
    while not frame_queue.empty():
        print(f"{datetime.datetime.now().isoformat()} | TFRecord: {tfrecord_id} - Tokens: {len(tokens)} - "
              f"Frames: {total_frames}")
        frames = frame_queue.get(timeout=600)
        total_frames += len(frames)
        tokens.extend(tokenize(model, frames, device))
        tfrecord_id += write_tfrecords(tokens, chunk_size, download_buffer_dir, save_dir, tfrecord_id,
                                       padding_token, cloud_storage_bucket)


def main():
    workers, bucket, prefix, tmp_dir, urls, min_duration, chunk_size, fps, startup_delay, batch_size, service_account_json, device, prefetch = parse_args()
    config_path = 'vqgan.gumbelf8.config.yml'
    model_path = 'sber.gumbelf8.ckpt'
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1WP6Li2Po8xYcQPGMpmaxIlI1yPB5lF5m', model_path, quiet=True)
    if not os.path.exists(config_path):
        gdown.download(f'https://drive.google.com/uc?id=1M7RvSoiuKBwpF-98sScKng0lsZnwFebR', config_path, quiet=True)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    conf = OmegaConf.load(config_path)
    padding_token = conf.model.params.n_embed
    resolution = conf.model.params.ddconfig.resolution
    video_ids = []
    durations = []
    model = load_vqgan(config_path, model_path)

    for url_path in os.listdir(urls):
        with open(f'{urls}/{url_path}', 'r') as f:
            json_load = json.load(f)
        video_ids.extend(json_load['id'])
        durations.extend(json_load['duration'])

    if not isinstance(video_ids[0], list):
        video_ids = [[video_id] for video_id in video_ids]
    else:
        durations = [np.sum(d) for d in durations]

    ids, duration = split_equal(video_ids, durations, workers, min_duration)

    split_chunk_count = 0
    split_video_count = 0
    split_video_duration = 0

    for i in range(len(ids)):
        buffer_chunk_count = len(ids[i])
        buffer_video_count = sum([len(__ids) for __ids in ids[i]])
        buffer_video_duration = sum(duration[i])

        print(f'Split: {i} - Chunks: {buffer_chunk_count} - Videos: {buffer_video_count} - '
              f'Duration: {buffer_video_duration}')

        split_chunk_count += buffer_chunk_count
        split_video_count += buffer_video_count
        split_video_duration += buffer_video_duration

    print(f'\nTotal Chunks: {split_chunk_count} - Total Videos: {split_video_count} - '
          f'Total Duration: {split_video_duration}\n')

    lock = multiprocessing.Lock()
    frame_queue = multiprocessing.Queue(prefetch)

    procs = [multiprocessing.Process(args=(work, worker_id, lock, resolution, tmp_dir, fps, batch_size, frame_queue),
                                     daemon=True, target=frame_worker) for worker_id, work in enumerate(ids)]
    for p in procs:
        p.start()

    return worker(model, chunk_size, prefix, tmp_dir, bucket, padding_token, service_account_json, torch.device(device),
                  frame_queue)


if __name__ == '__main__':
    main()
