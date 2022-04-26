import argparse
import datetime
import functools
import json
import multiprocessing
import os
import random
import subprocess
import threading
import time
import typing

import PIL
import PIL.Image
import cv2
import gdown
import numpy as np
import requests
import tensorflow as tf
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import youtube_dl
from google.cloud import storage
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help=f"Number of workers. Default is the number of CPU cores (={multiprocessing.cpu_count()})"
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
        "--webshare-api-key",
        type=str,
        help="API Key to https://www.webshare.io/"
    )
    args = parser.parse_args()
    return args.workers, args.bucket, args.prefix, args.tmp_dir, args.urls, args.min_duration, args.chunk_size, args.webshare_api_key, args.fps, args.startup_delay


def division_zero(x, y):
    return 0 if y == 0 else (x / y)


class Downloader:
    def __init__(self, max_try: int = 3, webshare_io_key: str = None):
        self.max_try = max_try
        self.webshare_io_key = webshare_io_key
        self.proxies = []

        self.update_proxy()

    def download(self, url: str, filename: str, use_proxy: bool):

        for _ in range(self.max_try):
            try:
                if use_proxy:
                    r = requests.get(url, stream=True, proxies=self.proxies)
                else:
                    r = requests.get(url, stream=True)

                with open(filename, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)

            except Exception as e:
                print(f'Download error with proxy: {use_proxy} \n error:', e)

                if use_proxy:
                    self.update_proxy()

            else:
                return True

        print('Retry exceeded for URL:', url)

        if os.path.exists(filename):
            os.remove(filename)

        return False

    def update_proxy(self):
        proxies = []
        _next = "/api/proxy/list/?page=1"
        while True:
            r = requests.get('https://proxy.webshare.io' + _next,
                             headers={"Authorization": f"Token {self.webshare_io_key}"})
            dump = r.json()

            if dump is None or 'next' not in dump:
                break

            _next = dump['next']
            proxies.extend([res for res in dump['results'] if res['valid']])

        random.shuffle(proxies)

        for p in proxies:
            username = p['username']
            password = p['password']
            proxy_address = p['proxy_address']
            ports = p['ports']['http']
            self.proxies.append({"http": f"http://{username}:{password}@{proxy_address}:{ports}",
                                 "https": f"http://{username}:{password}@{proxy_address}:{ports}"})


def frame_encoder(frame):
    feature = {'text': tf.train.Feature(int64_list=tf.train.Int64List(value=frame))}
    features = tf.train.Features(feature=feature)
    proto = tf.train.Example(features=features)
    proto = proto.SerializeToString()
    return proto


def split_equal(ids: list, duration: list, num: int, min_duration: int = 256):
    sort = sorted(zip(duration, ids))[::-1]

    ids_split = [[] for i in range(num)]
    duration_spit = [[] for i in range(num)]
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
def tokenize(model: GumbelVQ, frames: list, target_image_size: int, device: torch.device, batch_size: int):
    images = []
    with torch.no_grad():
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame)
            s = min(img.size)
            r = target_image_size / s
            s = (round(r * img.size[1]), round(r * img.size[0]))
            img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
            img = TF.center_crop(img, output_size=2 * [target_image_size])
            img = T.ToTensor()(img)
            images.append(img)
        batches = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size])
            batch = batch.to(device=device, non_blocking=True)
            batch, _, _ = model.encode(batch)
            batches.append(batch.detach())
        output = []
        for batch in batches:
            output.extend(batch.to(device='cpu:0', non_blocking=True).flatten().tolist())
    return output


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
    try:
        with cv2.VideoCapture(video_buffer_path) as video_cap:
            success, frame = video_cap.read()
        return success
    except:
        return False


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
        if not downloader.download(url, video_buffer_path, False):
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
    with cv2.VideoCapture(path) as video_cap:
        fps_split = division_zero(round(video_cap.get(cv2.CAP_PROP_FPS)), target_fps)
        while success:
            success, frame = video_cap.read()
            if frame_idx % fps_split == 0:
                frames.append(cv2.resize(frame, (target_image_size, target_image_size)))
            frame_idx += 1
    return frames


@functools.partial(try_except, default=0)
def write_tfrecords(tokens: typing.List[int], chunk_size: int, buffer_save_dir: str, save_dir: str, worker_id: int,
                    tfrecord_id: int, padding_token: int, cloud_storage_bucket):
    path = f"{buffer_save_dir}/{worker_id}.tfrecord"
    count = len(tokens)
    residual = count % chunk_size
    count -= residual
    added = 0

    for i in range(0, count, chunk_size):
        with tf.io.TFRecordWriter(path) as tf_writer:
            tf_writer.write(frame_encoder(tokens[i:i + chunk_size]))
        blob = cloud_storage_bucket.blob(f'{save_dir}/{worker_id:02d}_{tfrecord_id + added:07d}.tfrecord')
        blob.upload_from_filename(path)
        os.remove(path)
        added += 1
    residual_tokens = tokens[-residual:] + [padding_token]
    tokens.clear()
    tokens.extend(residual_tokens)
    return added


def worker(model: GumbelVQ,
           chunk_size: int,
           work: list,
           save_dir: str,
           target_fps: int,
           lock: threading.Lock,
           download_buffer_dir: str,
           webshare_io_key: str,
           bucket_name: str,
           padding_token: int,
           target_image_size: int,
           batch_size: int):
    torch.set_default_tensor_type('torch.FloatTensor')
    worker_id = xm.get_ordinal()
    youtube_base = 'https://www.youtube.com/watch?v='
    buffer_save_dir = download_buffer_dir

    cloud_storage_bucket = storage.Client().get_bucket(bucket_name)

    random.shuffle(work)

    youtube_getter = youtube_dl.YoutubeDL({'writeautomaticsub': False, 'ignore-errors': True, 'socket-timeout': 600})
    youtube_getter.add_default_info_extractors()
    device = xm.xla_device()
    model = model.to(device=device, non_blocking=True)

    downloader = Downloader(webshare_io_key=webshare_io_key)

    tfrecord_id = 0
    tokens = []
    for chunk_idx, wor in enumerate(work):
        for wor_idx, _wor in enumerate(wor):
            print(f"worker: {worker_id} chunk: {chunk_idx} video: {wor_idx}")
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
            tokens.extend(tokenize(model, frames, target_image_size, device, batch_size))
            tfrecord_id += write_tfrecords(tokens, chunk_size, buffer_save_dir, save_dir, worker_id, tfrecord_id,
                                           padding_token, cloud_storage_bucket)

    print(f'DONE worker: {worker_id}')


def main():
    workers, bucket, prefix, tmp_dir, urls, min_duration, chunk_size, webshare_api_key, fps, startup_delay = parse_args()
    config_path = 'vqgan.gumbelf8.config.yml'
    model_path = 'sber.gumbelf8.ckpt'
    batch_size = 48  # result from manual testing
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
    model = xmp.MpModelWrapper(model)

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

    def _exec_fn(rank: int, local_lock: multiprocessing.Lock):
        time.sleep(rank * startup_delay)
        print(f'Started Worker {rank} at {datetime.datetime.now()}')
        return worker(model, chunk_size, ids[rank], prefix, fps, local_lock, tmp_dir, webshare_api_key, bucket,
                      padding_token, resolution, batch_size)

    xmp.spawn(_exec_fn, nprocs=len(ids), start_method="fork", args=(lock,))


if __name__ == '__main__':
    main()
