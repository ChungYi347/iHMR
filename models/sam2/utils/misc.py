import os, sys
sys.path.append(os.path.abspath("./sam2"))

import torch
from tqdm import tqdm
from sam2.utils.misc import load_video_frames_from_video_file #, AsyncVideoFrameLoader #, _load_img_as_tensor,  # load_video_frames_from_jpg_images

import torchvision
from torchvision.transforms import InterpolationMode, ToTensor, Resize, Normalize as _Normalize
import torch.nn.functional as F
from PIL import Image
import numpy as np
from threading import Thread
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from util.preprocessing import generate_patch_image, load_img
import torch.nn as nn


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
        resize_size=None,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device
        self.resize_size = resize_size 

        if self.resize_size is not None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.to_tensor = ToTensor()
            self.transforms = torch.jit.script(
                nn.Sequential(
                    Resize((self.image_size, self.image_size)),
                    _Normalize(mean, std),
                )
            )

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size, self.resize_size
        )
        # import pdb; pdb.set_trace()
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        if self.resize_size:
            img = self.to_tensor(img)
            img = self.transforms(img)
        else:
            img -= self.img_mean
            img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)

def _resize_to_wh_chw(img_chw: torch.Tensor, target_w: int, target_h: int):
    """img_chw: (C,H,W) -> (C,target_h,target_w)"""
    return torchvision.transforms.functional.resize(
        img_chw,
        size=[target_h, target_w],  
        interpolation=InterpolationMode.BILINEAR
    )


def _fit_then_pad_to_square_1024(img_chw: torch.Tensor, final_size=1024, pad_value=0.0):
    C, H, W = img_chw.shape
    scale = min(final_size / float(H), final_size / float(W), 1.0)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    if new_h != H or new_w != W:
        img_chw = F.interpolate(img_chw.unsqueeze(0), size=(new_h, new_w),
                                mode='bilinear', align_corners=False).squeeze(0)
    pad_h = final_size - new_h
    pad_w = final_size - new_w
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    img_chw = F.pad(img_chw, pad, value=pad_value)
    return img_chw


def _load_img_as_tensor(img_path, image_size, resize_size=None):
    if resize_size is not None:
        # img_pil = Image.open(img_path)
        # video_width, video_height = img_pil.size  
        img_pil = load_img(img_path, order='BGR')
        video_width, video_height = img_pil.shape[1], img_pil.shape[0] 
        #  [   0    0 1080 1920] 1 0.0 False (1333, 750)
        img_pil, trans, inv_trans = generate_patch_image(img_pil, np.array([0, 0, img_pil.shape[1], img_pil.shape[0]]), 1, 0.0, False, (int(resize_size[1]), int(resize_size[0])))
        img_pil = np.clip(img_pil, 0, 255)

        # w, h = int(resize_size[0]), int(resize_size[1])
        # img_pil = img_pil.resize((w, h), resample=Image.BILINEAR)
        # img_pil = np.array(img_pil)[..., ::-1]
        normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        img_pil = normalize({"img": np.array(img_pil)})['img']
        _mean = torch.tensor([[[123.6750]], [[116.2800]], [[103.5300]]])
        _std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]])
        # img_pil = img_pil.transpose(2, 0, 1)
        img_pil = torch.from_numpy(img_pil).permute(2, 0, 1)
        img_pil = img_pil * _std + _mean
        img = torch.clamp(img_pil, 0, 255).to(torch.uint8).permute(1,2,0)
        img = np.array(img.cpu().numpy())
        # img = np.clip(np.round(img_pil, 4), 0, 255).astype(np.uint8).transpose(1,2,0)
        # img_pil = torch.from_numpy(img_pil).contiguous().float()
        # resize_model_tf = Resize(size=(w, h), interpolation=Image.BILINEAR, antialias=True)
        # img_pil = resize_model_tf(img_pil)
        # resize_tf = Resize(size=(image_size, image_size), interpolation=Image.BILINEAR, max_size=None, antialias=True)
        # img = resize_tf(img_pil) # / 255.0
        # img_pil = img_pil.permute(1, 2, 0)
        # img_pil = img_pil.transpose(1, 2, 0)
        # img_np = np.array(img_pil)                  # uint8
        # img = torch.from_numpy(img_np).permute(2, 0, 1) 
    else:
        img_pil = Image.open(img_path).convert("RGB")
        video_width, video_height = img_pil.size  
        img_pil = img_pil.resize((image_size, image_size), resample=Image.BILINEAR, reducing_gap=None)

        img_np = np.array(img_pil)                  # uint8
        img_np = img_np.astype(np.float32) / 255.0  # [0,1]
        img = torch.from_numpy(img_np).permute(2, 0, 1)  # (C,H,W)

    return img, video_height, video_width


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    resize_size=None,
    frame_ranges=None,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    elif isinstance(video_path, list):
        frame_names = [
            p
            for p in video_path
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    else:
        raise NotImplementedError(
            "Only JPEG frames are supported at this moment. For video files, you may use "
            "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg."
        )

    
    frame_names.sort(key=lambda p: int(os.path.splitext(p.replace('img_', '').replace('image_', ''))[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    if frame_ranges is not None:
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names if frame_name in frame_ranges]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
            resize_size,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size, resize_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_png_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    resize_size=None,
    frame_ranges=None,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".png", ".PNG"]
        ]
    elif isinstance(video_path, list):
        frame_names = [
            p
            for p in video_path
            if os.path.splitext(p)[-1] in [".png", ".PNG"]
        ]
    else:
        raise NotImplementedError(
            "Only JPEG frames are supported at this moment. For video files, you may use "
            "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg."
        )
    
    if 'bedlam' in video_path:
        folder_path = video_path.split('/')[-1] + "_"
        frame_names.sort(key=lambda p: int(os.path.splitext(p.replace(folder_path, ''))[0]))
    else:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
            resize_size,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size, resize_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width

def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    resize_size=None,
    frame_ranges=None,
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    is_png_path = False
    
    if os.path.isdir(video_path):
        is_png_path = any(file.lower().endswith(".png") or file.lower().endswith(".PNG") for file in os.listdir(video_path))

    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
            resize_size=resize_size,
        )
    elif is_str and os.path.isdir(video_path) and is_png_path:
        return load_video_frames_from_png_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
            resize_size=resize_size,
            frame_ranges=frame_ranges,
        )
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
            resize_size=resize_size,
            frame_ranges=frame_ranges,
        )
    else:
        raise NotImplementedError(
            "Only MP4 video and JPEG folder are supported at this moment"
        )