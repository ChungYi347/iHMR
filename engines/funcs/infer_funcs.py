import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb, get_colors_rgb_tracking_ids
from utils.evaluation import cal_3d_position_error, match_2d_greedy, get_matching_dict, compute_prf1, vectorize_distance, calculate_iou, select_and_align
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.process_batch import prepare_batch
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, j2ds_to_bboxes_xywh
import time
import cv2
import trimesh

# from ultralytics import YOLO
import sys
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ONNX-YOLOv8-Object-Detection"))
from yolov8 import YOLOv8
from models.geometry import axis_angle_to_matrix
import csv
from scenedetect.detectors import ContentDetector
from utils.constants import H36M_EVAL_JOINTS
import json
from PIL import Image

def init_color_ref(n, seed=12345):
    """Sample N colors in HSV and convert them to RGB."""
    rand_state = np.random.RandomState(seed)
    rand_hsv = rand_state.rand(n, 3)
    rand_hsv[:, 1:] = 1 - rand_hsv[:, 1:] * 0.3
    color_ref = hsv2rgb(rand_hsv.T)

    return color_ref
def hsv2rgb(hsv):
    """Convert a tuple from HSV to RGB."""
    h, s, v = hsv
    vals = np.tile(v, [3] + [1] * v.ndim)
    vals[1:] *= 1 - s[None]

    h[h > (5 / 6)] -= 1
    diffs = np.tile(h, [3] + [1] * h.ndim) - (np.arange(3) / 3).reshape(
        3, *[1] * h.ndim
    )
    max_idx = np.abs(diffs).argmin(0)

    final_rgb = np.zeros_like(vals)

    for i in range(3):
        tmp_d = diffs[i] * (max_idx == i)
        dv = tmp_d * 6 * s * v
        vals[1] += np.maximum(0, dv)
        vals[2] += np.maximum(0, -dv)

        final_rgb += np.roll(vals, i, axis=0) * (max_idx == i)

    return final_rgb.transpose(*list(np.arange(h.ndim) + 1), 0)

def inference(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    if hasattr(model, 'human_model'):
        smpl_layer = model.human_model
    else:
        smpl_layer = model.smplx
    # yolo = YOLO("data/pretrain/yolov8x.pt")
    detector = YOLOv8('weights/yolov8m.onnx', conf_thres=0.1, iou_thres=0.5)

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')
    
    detect_conf = 0.3
    for itr, (samples, targets) in enumerate(infer_dataloader):
        if model.__class__.__name__ == "PHMR":
            inputs = []
            for target in targets:
                img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
                
                # detection = yolo(target['img_path'], verbose=False, conf=detect_conf, classes=0)
                # boxes = detection[0].boxes.data.cpu()
                detected = detector(img)
                boxes = detected[0][detected[2] == 0]
                # boxes = detected[0][detected[2] == 0] / model_cfg['MODEL']['IMAGE_SIZE']

                # _img = cv2.imread(targets[0]['img_path'])[:,:,::-1]
                # ori_img_szie = _img.shape

                # boxes[:, 0] = boxes[:, 0] / ori_img_szie[0] * img.shape[0]
                # boxes[:, 1] = boxes[:, 1] / ori_img_szie[1] * img.shape[1]
                # boxes[:, 2] = boxes[:, 3] / ori_img_szie[0] * img.shape[0]
                # boxes[:, 3] = boxes[:, 3] / ori_img_szie[1] * img.shape[1]
                img_name = target['img_path'].split('/')[-1].split('.')[0]
                # img_size = img.shape
                # ori_img = img
                # ori_img[img_size[0]:,:,:] = 255
                # ori_img[:,img_size[1]:,:] = 255
                # ori_img[img_size[0]:,img_size[1]:,:] = 255
                # ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                # pred_box_img = vis_boxes(ori_img.copy(), boxes, color = (255,0,255))[:img_size[0],:img_size[1]]
                img = np.ascontiguousarray(img)
                # for box in boxes.numpy():
                for box in boxes:
                    box = box.astype(np.int32)
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)

                cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), img)

                boxes = torch.tensor(boxes)

                _input = {'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}
                inputs.append(_input)
            # inputs = [{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}]

            batch = prepare_batch(inputs, img_size=896, interaction=False)

            # samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            with torch.no_grad():    
                # outputs = model(samples, targets)
                outputs = model(batch, targets)
        elif model.__class__.__name__ == "Sam2Model":
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(samples[0].device)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(samples[0].device)
            img = samples[0] * std + mean
            img = torch.clamp(img, 0, 255).to(torch.uint8)
    
            detected = detector(img.cpu().numpy().transpose(1,2,0))
            boxes = detected[0][detected[2] == 0]
            targets[0]['boxes'] = box_xyxy_to_cxcywh(torch.tensor(boxes)) / model.input_size 
            targets[0]['cam_intrinsics'] = model.cam_intrinsics[None]

            with torch.no_grad():    
                outputs = model(samples, targets)
        else:
            img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
            detected = detector(img)
            boxes = detected[0][detected[2] == 0]
            targets[0]['boxes'] = box_xyxy_to_cxcywh(torch.tensor(boxes).to(samples[0].device)) / model.input_size 
            samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            with torch.no_grad():    
                outputs = model(samples, targets)

        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            #pred
            if model.__class__.__name__ == "PHMR":
                pred_verts = outputs[idx]['vertices'].detach().cpu().numpy()
            elif model.__class__.__name__ == "Sam2Model":
                pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
            else:
                select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
                pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            ori_img[img_size[0]:,:,:] = 255
            ori_img[:,img_size[1]:,:] = 255
            ori_img[img_size[0]:,img_size[1]:,:] = 255

            if model.__class__.__name__ == "PHMR":
                ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                colors = get_colors_rgb(len(pred_verts))
                # pred_verts = smplx_out.vertices
                # smpl_layer = smpl
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = batch[idx]['cam_int_original'][0].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'tt_{img_name}.png'), pred_mesh_img)
            elif model.__class__.__name__ == "Sam2Model":
                ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                colors = get_colors_rgb(len(pred_verts))
                # pred_verts = smplx_out.vertices
                # smpl_layer = smpl
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
            else:
                ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)[:img_size[0],:img_size[1]]

                # colors = get_colors_rgb(len(pred_verts), tracking_ids=outputs['track_ids'])
                colors = get_colors_rgb(len(pred_verts))
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                
                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map

                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                    scale_map = scale_map,
                                                    conf_thresh = model.sat_cfg['conf_thresh'],
                                                    patch_size=28)[:img_size[0],:img_size[1]]

                pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()
                joint_mapper = np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],dtype=np.int32)
                pred_j2ds = pred_j2ds[:, joint_mapper]

                joint_regressor_extra = "../4D-Humans/data/SMPL_to_J19.pkl"
                regressor = torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32)
                from smplx.lbs import vertices2joints
                extra_joints = vertices2joints(regressor, torch.tensor(pred_verts))
                j2ds_homo = torch.matmul(extra_joints, outputs['pred_intrinsics'].transpose(2,3).cpu())
                extra_joints = (j2ds_homo[..., :2] / (j2ds_homo[..., 2, None] + 1e-6))

                kps_img = ori_img.copy()
                # for pred_j2d in extra_joints[0].cpu().numpy():
                for n, pred_j2d in enumerate(pred_j2ds):
                    for j, (x, y) in enumerate(pred_j2d):
                        cv2.circle(kps_img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)

                        if j == 0:
                            cv2.putText(kps_img, str(n), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0,255,255), 1, cv2.LINE_AA)
                kps_img = kps_img[:pred_mesh_img.shape[0], :pred_mesh_img.shape[1]]

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                #                                                                             np.hstack([pred_scale_img, sat_img])]))
                cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                                                                                            np.hstack([kps_img, sat_img])]))


        progress_bar.update(1)
    

def rgb_hist(img, bins=32):
    """
    Compute RGB histogram for an image.
    
    Args:
        img (PIL.Image or np.ndarray): RGB image.
        bins (int): Number of histogram bins per channel.
    
    Returns:
        np.ndarray: Concatenated histogram of shape (3 * bins,)
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    hist_list = []
    for c in range(3):  # R, G, B
        hist, _ = np.histogram(img[..., c], bins=bins, range=(0, 255), density=True)
        hist_list.append(hist)

    return np.concatenate(hist_list)

def phash(image, hash_size=8, highfreq_factor=4):
    """
    Perceptual hash (64비트 boolean 배열).
    image: PIL.Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # 1) 그레이스케일 + 리사이즈
    img = image.convert("L").resize(
        (hash_size * highfreq_factor, hash_size * highfreq_factor),
        Image.BILINEAR
    )
    pixels = np.array(img, dtype=np.float32)

    # 2) DCT에 준하는 주파수 특징: 간단히 2D FFT 절댓값 사용
    dct = np.fft.fft2(pixels)
    dct = np.abs(dct)

    # 3) 저주파 블록
    dctlow = dct[:hash_size, :hash_size]

    # 4) DC 제외 중앙값 기준 이진화
    med = np.median(dctlow[1:, 1:])
    return (dctlow > med)

def hamming(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    return int(np.count_nonzero(a.flatten() != b.flatten()))


def is_cut(prev_img, curr_img, thresh_phash=18, thresh_chi2=0.03):
    ph1 = phash(prev_img)   # 64bit (8x8)
    ph2 = phash(curr_img)
    ham = hamming(ph1, ph2)
    if ham > thresh_phash:
        return True

    h1 = rgb_hist(prev_img, bins=32)  # density=True
    h2 = rgb_hist(curr_img, bins=32)
    chi2 = 0.5 * np.sum((h1 - h2)**2 / (h1 + h2 + 1e-10))
    return chi2 > thresh_chi2

def eval_posetrack(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None, args=None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    if hasattr(model, 'human_model'):
        smpl_layer = model.human_model
    else:
        smpl_layer = model.smplx
    # yolo = YOLO("data/pretrain/yolov8x.pt")
    detector = YOLOv8('weights/yolov8m.onnx', conf_thres=0.1, iou_thres=0.5)

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')

    save_path = os.path.join(results_save_path, 'pred.json')
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

    shot_detector = ContentDetector(threshold=50.0, min_scene_len=3)
    frame_count = 0
    
    model.query_bank = []
    if hasattr(model, 'module'):
        model.module.query_bank = []
        
    detect_conf = 0.3
    times = []
    prev_frame = None
    for itr, (samples, targets) in enumerate(infer_dataloader):
        model.tracker.frame_id = itr
        model.img_seq_idx = itr
        
        # cur_frame = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
        # if prev_frame is not None and is_cut(prev_frame, cur_frame):
        #     print('hello')
        #     model.tracker.reset()           

        if hasattr(model, 'module'):
            model.img_seq_idx = itr
        if model.__class__.__name__ == "PHMR":
            inputs = []
            
            for target in targets:
                img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
                
                # detection = yolo(target['img_path'], verbose=False, conf=detect_conf, classes=0)
                # boxes = detection[0].boxes.data.cpu()
                detected = detector(img)
                boxes = detected[0][detected[2] == 0]
                # boxes = detected[0][detected[2] == 0] / model_cfg['MODEL']['IMAGE_SIZE']

                # _img = cv2.imread(targets[0]['img_path'])[:,:,::-1]
                # ori_img_szie = _img.shape

                # boxes[:, 0] = boxes[:, 0] / ori_img_szie[0] * img.shape[0]
                # boxes[:, 1] = boxes[:, 1] / ori_img_szie[1] * img.shape[1]
                # boxes[:, 2] = boxes[:, 3] / ori_img_szie[0] * img.shape[0]
                # boxes[:, 3] = boxes[:, 3] / ori_img_szie[1] * img.shape[1]
                img_name = target['img_path'].split('/')[-1].split('.')[0]
                # img_size = img.shape
                # ori_img = img
                # ori_img[img_size[0]:,:,:] = 255
                # ori_img[:,img_size[1]:,:] = 255
                # ori_img[img_size[0]:,img_size[1]:,:] = 255
                # ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                # pred_box_img = vis_boxes(ori_img.copy(), boxes, color = (255,0,255))[:img_size[0],:img_size[1]]
                img = np.ascontiguousarray(img)
                # for box in boxes.numpy():
                for box in boxes:
                    box = box.astype(np.int32)
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)

                cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), img)

                boxes = torch.tensor(boxes)

                _input = {'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}
                inputs.append(_input)
            # inputs = [{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}]

            batch = prepare_batch(inputs, img_size=896, interaction=False)

            # samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            with torch.no_grad():    
                # outputs = model(samples, targets)
                outputs = model(batch, targets)
        elif model.__class__.__name__ == "Sam2Model":
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(samples[0].device)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(samples[0].device)
            img = samples[0] * std + mean
            img = torch.clamp(img, 0, 255).to(torch.uint8)
    
            detected = detector(img.cpu().numpy().transpose(1,2,0))
            boxes = detected[0][detected[2] == 0]
            targets[0]['boxes'] = box_xyxy_to_cxcywh(torch.tensor(boxes)) / model.input_size 
            targets[0]['cam_intrinsics'] = model.cam_intrinsics[None]

            with torch.no_grad():    
                outputs = model(samples, targets)
        elif model.__class__.__name__ == "ImageModel":
            # if itr == 0:
            #     img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
            #     detected = detector(img)
            #     boxes = detected[0][detected[2] == 0]

            #     samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            #     with torch.no_grad():    
            #         outputs = model(samples, targets, boxes)
            # else:
            #     samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            #     with torch.no_grad():    
            #         outputs = model(samples, targets)
            # img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
            # shots = shot_detector.process_frame(frame_count, img)
            # frame_count += 1
            # is_new_shot = len(shots) > 0 if frame_count > 1 else False
            # print(targets[0]['img_path'], shots, is_new_shot)
            # model.tracker.reset()
            # if is_new_shot:
            #     model.tracker.reset()
            #     print(model.tracker.next_id)
            #     print(targets[0]['img_path'], shots)
            
            samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            with torch.no_grad():    
                start_time = time.time()
                outputs = model(samples, targets)
                end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
        else:
            samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            with torch.no_grad():    
                outputs = model(samples, targets)

            # prev_frame = cur_frame

        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            # # # #pred
            if model.__class__.__name__ == "PHMR":
                pred_verts = outputs[idx]['vertices'].detach().cpu().numpy()
            elif model.__class__.__name__ == "Sam2Model":
                pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
            elif model.__class__.__name__ == "VideoModel":
                pred_verts = outputs['pred_verts'].detach().cpu().numpy()
            else:
                # select_queries_idx = torch.where(outputs['pred_confs'][idx] > model.tracker.conf_thresh)[0]
                # pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()
                pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
            
            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            ori_img[img_size[0]:,:,:] = 255
            ori_img[:,img_size[1]:,:] = 255
            ori_img[img_size[0]:,img_size[1]:,:] = 255

            if model.__class__.__name__ == "PHMR":
                ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                colors = get_colors_rgb(len(pred_verts))
                # pred_verts = smplx_out.vertices
                # smpl_layer = smpl
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = batch[idx]['cam_int_original'][0].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'tt_{img_name}.png'), pred_mesh_img)
            elif model.__class__.__name__ == "Sam2Model":
                ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                colors = get_colors_rgb(len(pred_verts))
                # pred_verts = smplx_out.vertices
                # smpl_layer = smpl
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
            else:
                ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
                # sat_img = vis_sat(ori_img.copy(),
                #                     input_size = model.input_size,
                #                     patch_size = 14,
                #                     sat_dict = outputs['sat'],
                #                     bid = idx)[:img_size[0],:img_size[1]]

                if model.__class__.__name__ == "VideoModel":
                    colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
                else:
                    colors = get_colors_rgb(len(pred_verts))
                colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
                


                color_ref = init_color_ref(2000)
                colors = color_ref[outputs['track_ids'][0]]
                # colors = get_colors_rgb(len(pred_verts))
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts[idx],
                                            # verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                    
                # if 'enc_outputs' not in outputs:
                #     pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
                # else:
                #     enc_out = outputs['enc_outputs']
                #     h, w = enc_out['hw'][idx]
                #     flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                #     ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                #     xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                #     scale_map = torch.zeros((h,w,2))
                #     scale_map[ys,xs] = flatten_map

                #     pred_scale_img = vis_scale_img(img = ori_img.copy(),
                #                                     scale_map = scale_map,
                #                                     conf_thresh = model.sat_cfg['conf_thresh'],
                #                                     patch_size=28)[:img_size[0],:img_size[1]]

                # pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                # pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                #                                                                             np.hstack([pred_scale_img, sat_img])]))

                # pred_confs = outputs['ori_pred_confs'][idx]
                # pred_boxes = outputs['ori_pred_boxes'][idx][pred_confs[:, 0] > conf_thresh].detach().cpu() * model.input_size
                # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
                # pred_boxes = outputs['pred_boxes'][idx].detach().cpu() * model.input_size
                # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

                # box_img = ori_img.copy()
                # for box_idx, bbox in enumerate(pred_boxes):
                #     x1, y1, x2, y2 = bbox.int().tolist()
                #     # kfx1, kfy1, kfx2, kfy2 = model.tracker.tracks[tid]['kf'].get_xyxy().tolist()
                #     cv2.rectangle(box_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                #     # cv2.rectangle(pred_mesh_img, (int(kfx1), int(kfy1)), (int(kfx2), int(kfy2)), (0, 255, 0), 4)
                #     cv2.putText(
                #         box_img,
                #         f"{box_idx}",
                #         (x1, y1 - 5),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.8,
                #         (255, 0, 255),
                #         2,
                #         cv2.LINE_AA
                #     )
                # box_img = box_img[:img_size[0],:img_size[1]]

                # box_img = ori_img.copy()
                # pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
                # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
                # _pred_boxes = outputs['pred_boxes'][idx].detach().cpu() 
                # _pred_boxes = box_cxcywh_to_xyxy(_pred_boxes) * model.input_size

                # pred_mesh_img = vis_boxes(pred_mesh_img, pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]
                # pred_mesh_img = vis_boxes(pred_mesh_img, _pred_boxes, color = (0,0,255))[:img_size[0],:img_size[1]]
                # for box_idx, tid in enumerate(outputs['active_ids']):
                #     x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
                #     # kfx1, kfy1, kfx2, kfy2 = model.tracker.tracks[tid]['kf'].get_xyxy().tolist()
                #     cv2.rectangle(pred_mesh_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                #     # cv2.rectangle(pred_mesh_img, (int(kfx1), int(kfy1)), (int(kfx2), int(kfy2)), (0, 255, 0), 4)
                #     cv2.putText(
                #         pred_mesh_img,
                #         f"{tid}",
                #         (x1, y1 - 5),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.8,
                #         (255, 0, 255),
                #         2,
                #         cv2.LINE_AA
                #     )
                # pred_mesh_img = vis_boxes(pred_mesh_img, pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                # # select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
                # # ori_pred_boxes = outputs['ori_pred_boxes'][idx][select_queries_idx]
                # # ori_pred_boxes = box_cxcywh_to_xyxy(ori_pred_boxes) * model.input_size
                # # pred_mesh_img = vis_boxes(pred_mesh_img, ori_pred_boxes, color = (0,255,0))[:img_size[0],:img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), pred_mesh_img)
                # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), box_img)
                

            # print(len(outputs['pred_verts'][0]))
            # print(model.tracker.active_ids)
            ori_img_size = targets[0]['ori_img_size']
            img_size = targets[0]['img_size']
            def save_tracking_to_csv(frame_id, tracker, file_path, img_size, ori_img_size, overlap_iou=0.8):
                selected_tids = set(tracker.select_tracks_by_self_iou(overlap_iou))
                if not selected_tids:
                    return
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for tid in tracker.active_ids:
                        if tid not in selected_tids:
                            continue
                        tinfo = tracker.tracks[tid]
                        x1, y1, x2, y2 = np.array(tinfo["box"].tolist())
                        x1 = int(x1 * ori_img_size[0] / img_size[0])
                        x2 = int(x2 * ori_img_size[0] / img_size[0])
                        y1 = int(y1 * ori_img_size[1] / img_size[1])
                        y2 = int(y2 * ori_img_size[1] / img_size[1])
                        # writer.writerow([frame_id, tid, x1, y1, x2-x1, y2-y1, 1, 0, 0, 0])
                        writer.writerow([frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0])
            # def save_tracking_to_csv(frame_id, tracker, file_path=save_path):
            #     with open(file_path, "a", newline="") as f:
            #         writer = csv.writer(f)
            #         for tid, tinfo in tracker.tracks.items():
            #             x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) 
            #             # x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) / model.input_size
            #             x1, x2 = int(x1 * ori_img_size[0] / img_size[0]), int(x2 * ori_img_size[0] / img_size[0])
            #             y1, y2 = int(y1 * ori_img_size[1] / img_size[1]), int(y2 * ori_img_size[1] / img_size[1])
            #             row = [frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0]
            #             writer.writerow(row)
            # save_tracking_to_csv(itr+1, model.tracker)
            # import pdb; pdb.set_trace()
            save_tracking_to_csv(
                frame_id=itr+1,
                tracker=model.tracker,
                file_path=save_path,
                img_size=img_size,
                ori_img_size=ori_img_size,
                overlap_iou=0.8
            )
        time_file_path = os.path.dirname(save_path)
        with open(f"{time_file_path}/inference_times.json", "w") as f:
            json.dump(times, f, indent=4)

        progress_bar.update(1)

def eval_3dpw_track(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None, args=None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    if hasattr(model, 'human_model'):
        smpl_layer = model.human_model
    else:
        smpl_layer = model.smplx
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    # yolo = YOLO("data/pretrain/yolov8x.pt")
    # detector = YOLOv8('weights/yolov8m.onnx', conf_thres=0.1, iou_thres=0.5)

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')

    save_path = os.path.join(results_save_path, 'pred.json')
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

    shot_detector = ContentDetector(threshold=50.0, min_scene_len=3)
    frame_count = 0
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    # store arrays per sample then concatenate once at the end
    mve, mpjpe, pa_mpjpe, pa_mve = [], [], [], []

    model.query_bank = []
    if hasattr(model, 'module'):
        model.module.query_bank = []
         
    detect_conf = 0.3
    for itr, (samples, targets) in enumerate(infer_dataloader):
        model.tracker.frame_id = itr
        model.img_seq_idx = itr
        if hasattr(model, 'module'):
            model.img_seq_idx = itr

        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
            outputs = model(samples, targets)

        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
            # print(img_name)

            if itr > 110:
                # # # #pred
                if model.__class__.__name__ == "PHMR":
                    pred_verts = outputs[idx]['vertices'].detach().cpu().numpy()
                elif model.__class__.__name__ == "Sam2Model":
                    pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
                elif model.__class__.__name__ == "VideoModel":
                    pred_verts = outputs['pred_verts'].detach().cpu().numpy()
                else:
                    select_queries_idx = torch.where(outputs['pred_confs'][idx] > model.tracker.conf_thresh)[0]
                    pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()
                
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img[img_size[0]:,:,:] = 255
                ori_img[:,img_size[1]:,:] = 255
                ori_img[img_size[0]:,img_size[1]:,:] = 255

                if model.__class__.__name__ == "PHMR":
                    ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                    colors = get_colors_rgb(len(pred_verts))
                    # pred_verts = smplx_out.vertices
                    # smpl_layer = smpl
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                                verts = pred_verts,
                                                smpl_faces = smpl_layer.faces,
                                                cam_intrinsics = batch[idx]['cam_int_original'][0].reshape(3,3).detach().cpu(),
                                                colors=colors)[:img_size[0],:img_size[1]]
                    cv2.imwrite(os.path.join(results_save_path, f'tt_{img_name}.png'), pred_mesh_img)
                elif model.__class__.__name__ == "Sam2Model":
                    ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                    colors = get_colors_rgb(len(pred_verts))
                    # pred_verts = smplx_out.vertices
                    # smpl_layer = smpl
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                                verts = pred_verts,
                                                smpl_faces = smpl_layer.faces,
                                                cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                                colors=colors)[:img_size[0],:img_size[1]]
                    cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
                else:
                    ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
                    # sat_img = vis_sat(ori_img.copy(),
                    #                     input_size = model.input_size,
                    #                     patch_size = 14,
                    #                     sat_dict = outputs['sat'],
                    #                     bid = idx)[:img_size[0],:img_size[1]]

                    if model.__class__.__name__ == "VideoModel":
                        colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
                        pred_verts = pred_verts[0]
                    else:
                        colors = get_colors_rgb(len(pred_verts))
                    colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
                    # colors = get_colors_rgb(len(pred_verts))
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                                # verts = pred_verts[idx],
                                                verts = pred_verts,
                                                smpl_faces = smpl_layer.faces,
                                                cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                                colors=colors)[:img_size[0],:img_size[1]]
                        
                    # if 'enc_outputs' not in outputs:
                    #     pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
                    # else:
                    #     enc_out = outputs['enc_outputs']
                    #     h, w = enc_out['hw'][idx]
                    #     flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    #     ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    #     xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    #     scale_map = torch.zeros((h,w,2))
                    #     scale_map[ys,xs] = flatten_map

                    #     pred_scale_img = vis_scale_img(img = ori_img.copy(),
                    #                                     scale_map = scale_map,
                    #                                     conf_thresh = model.sat_cfg['conf_thresh'],
                    #                                     patch_size=28)[:img_size[0],:img_size[1]]

                    # pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                    # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                    # pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                    # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                    #                                                                             np.hstack([pred_scale_img, sat_img])]))

                    pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
                    # print(ori_img.shape, img_size)
                    # print(pred_boxes)
                    pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
                    # print(pred_boxes)
                    # pred_boxes = outputs['pred_boxes'][idx].detach().cpu() * model.input_size
                    # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

                    for box_idx, tid in enumerate(outputs['active_ids']):
                        x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
                        # print(itr, tid, x1, y1, x2, y2)
                        # kfx1, kfy1, kfx2, kfy2 = model.tracker.tracks[tid]['kf'].get_xyxy().tolist()

                        cv2.rectangle(pred_mesh_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        # cv2.rectangle(pred_mesh_img, (int(kfx1), int(kfy1)), (int(kfx2), int(kfy2)), (0, 255, 0), 4)
                        cv2.putText(
                            pred_mesh_img,
                            f"{tid}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA
                        )
                    pred_mesh_img = vis_boxes(pred_mesh_img, pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                    # select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
                    # ori_pred_boxes = outputs['ori_pred_boxes'][idx][select_queries_idx]
                    # ori_pred_boxes = box_cxcywh_to_xyxy(ori_pred_boxes) * model.input_size
                    # pred_mesh_img = vis_boxes(pred_mesh_img, ori_pred_boxes, color = (0,255,0))[:img_size[0],:img_size[1]]
                    print(img_name)
                    cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), pred_mesh_img)
                    import pdb; pdb.set_trace()

            # pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
            # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
            # for box_idx, tid in enumerate(outputs['active_ids']):
            #     x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
            #     print(itr, tid, x1, y1, x2, y2)

            # print(len(outputs['pred_verts'][0]))
            # print(model.tracker.active_ids)
            ori_img_size = targets[0]['ori_img_size']
            img_size = targets[0]['img_size']
            def save_tracking_to_csv(frame_id, tracker, file_path, img_size, ori_img_size, overlap_iou=0.8):
                # selected_tids = set(tracker.select_tracks_by_self_iou(overlap_iou))
                # if frame_id >= 117:
                # print(tracker.active_ids)
                # if not selected_tids:
                #     return
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for tid in tracker.active_ids:
                        if tid not in [0, 1]:
                            continue
                        # if tid not in selected_tids:
                        #     continue
                        tinfo = tracker.tracks[tid]
                        x1, y1, x2, y2 = np.array(tinfo["box"].tolist())
                        # print(x1,y1,x2,y2)
                        x1 = int(x1 * ori_img_size[0] / img_size[0] )
                        x2 = int(x2 * ori_img_size[0] / img_size[0] )
                        y1 = int(y1 * ori_img_size[1] / img_size[1] )
                        y2 = int(y2 * ori_img_size[1] / img_size[1] )
                        # x1 = int(x1 * ori_img_size[0] / self.model.input_size )
                        # x2 = int(x2 * ori_img_size[0] / self.model.input_size )
                        # y1 = int(y1 * ori_img_size[1] / self.model.input_size )
                        # y2 = int(y2 * ori_img_size[1] / self.model.input_size )
                        # x1 = int(x1 )
                        # x2 = int(x2 )
                        # y1 = int(y1 )
                        # y2 = int(y2 )
                        writer.writerow([frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0])
                        # writer.writerow([frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0])
            # def save_tracking_to_csv(frame_id, tracker, file_path=save_path):
            #     with open(file_path, "a", newline="") as f:
            #         writer = csv.writer(f)
            #         for tid, tinfo in tracker.tracks.items():
            #             x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) 
            #             # x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) / model.input_size
            #             x1, x2 = int(x1 * ori_img_size[0] / img_size[0]), int(x2 * ori_img_size[0] / img_size[0])
            #             y1, y2 = int(y1 * ori_img_size[1] / img_size[1]), int(y2 * ori_img_size[1] / img_size[1])
            #             row = [frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0]
            #             writer.writerow(row)
            # save_tracking_to_csv(itr+1, model.tracker)
            # import pdb; pdb.set_trace()
            save_tracking_to_csv(
                frame_id=itr+1,
                tracker=model.tracker,
                file_path=save_path,
                img_size=img_size,
                ori_img_size=ori_img_size,
                overlap_iou=0.8
            )
        
            # per-batch aggregators
            batch_count = []
            batch_miss_count = []
            batch_fp = []
            batch_mve = []
            batch_mpjpe = []
            batch_pa_mve = []
            batch_pa_mpjpe = []

            bs = len(targets)
            for idx in range(bs):
                batch_count.append(0)
                batch_miss_count.append(0)
                batch_fp.append(0)
                # avoid empty gather
                sample_mve = [float('inf')]
                sample_pa_mve = [float('inf')]
                sample_mpjpe = [float('inf')]
                sample_pa_mpjpe = [float('inf')]

                # gt 
                gt_verts = targets[idx]['verts'].to(device=cur_device)
                gt_transl = targets[idx]['transl'].to(device=cur_device)
                gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:,None,:], smpl2h36m_regressor]) + gt_transl[:,None,:]

                gt_verts = gt_verts.cpu().numpy()
                gt_j3ds = gt_j3ds.cpu().numpy()
                gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]

                select_queries_idx = torch.where(outputs['pred_confs'][idx] > model.tracker.conf_thresh)[0]
                pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
                pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
                pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]
                
                pred_verts = pred_verts.cpu().numpy()
                pred_j3ds = pred_j3ds.cpu().numpy()
                pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]

                # matching
                matched_verts_idx = []
                assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
                greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds)  # tuples are (idx_pred_kps, idx_gt_kps)
                matchDict, falsePositive_count = get_matching_dict(greedy_match)

                if model.__class__.__name__ == "PHMR" or model.__class__.__name__ == "Sam2Model":
                    gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
                    gtIdxs = np.arange(len(gt_j3ds))
                    for gtIdx in gtIdxs:
                        gt_verts_list.append(gt_verts[gtIdx])
                        gt_joints_list.append(gt_j3ds[gtIdx])
                        pred_joints_list.append(pred_j3ds[gtIdx])
                        pred_verts_list.append(pred_verts[gtIdx])
                        matched_verts_idx.append(gtIdx)
                else:
                    # align with matching result
                    gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
                    gtIdxs = np.arange(len(gt_j3ds))
                    miss_flag = []
                    for gtIdx in gtIdxs:
                        gt_verts_list.append(gt_verts[gtIdx])
                        gt_joints_list.append(gt_j3ds[gtIdx])
                        if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                            miss_flag.append(1)
                            pred_verts_list.append([])
                            pred_joints_list.append([])
                        else:
                            miss_flag.append(0)
                            pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                            pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                            matched_verts_idx.append(matchDict[str(gtIdx)])

                # compute 3D errors
                for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                    batch_count[-1] += 1

                    if model.__class__.__name__ != "PHMR" and model.__class__.__name__ != "Sam2Model":
                        if miss_flag[i] == 1:
                            batch_miss_count[-1] += 1
                            continue

                    gt3d = gt3d.reshape(-1, 3)
                    pred3d = pred.reshape(-1, 3)
                    gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                    pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                    gt_pelvis = gt3d[[0],:].copy()
                    pred_pelvis = pred3d[[0],:].copy()

                    gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                    gt3d_verts = (gt3d_verts - gt_pelvis).copy()
                    
                    pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                    pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                    # joints
                    # reconstruction_error
                    error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                    sample_mpjpe.append(float(error_j))
                    sample_pa_mpjpe.append(float(pa_error_j))
                    # vertices
                    error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                    sample_mve.append(float(error_v))
                    sample_pa_mve.append(float(pa_error_v))

                # counting and visualization
                step += 1
                batch_fp[-1] += falsePositive_count

                # stash per-sample arrays (with leading inf)
                batch_mve.append(np.array(sample_mve))
                batch_pa_mve.append(np.array(sample_pa_mve))
                batch_mpjpe.append(np.array(sample_mpjpe))
                batch_pa_mpjpe.append(np.array(sample_pa_mpjpe))
                # print(sample_mve, sample_pa_mve, sample_mpjpe, sample_pa_mpjpe)

                # img_idx = step + accelerator.process_index*len(infer_dataloader)*bs
                # if vis and (img_idx%vis_step == 0) and len(matched_verts_idx) > 0:
                #     img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                #     ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                #     ori_img = pad_img(ori_img, model.input_size)

                #     print(sample_mve, sample_pa_mve, sample_mpjpe, sample_pa_mpjpe)

                #     selected_verts = pred_verts[matched_verts_idx]
                #     colors = get_colors_rgb(len(selected_verts))
                #     if model.__class__.__name__ == "PHMR":
                #         mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                     verts = selected_verts,
                #                                     smpl_faces = smpl_layer.faces,
                #                                     colors = colors,
                #                                     cam_intrinsics = outputs[idx]['cam_int_original'][matched_verts_idx].detach().cpu())
                #         full_img = np.hstack([ori_img, mesh_img])
                #     elif model.__class__.__name__ == "Sam2Model":
                #         mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                     verts = selected_verts,
                #                                     smpl_faces = smpl_layer.faces,
                #                                     colors = colors,
                #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                #         full_img = np.hstack([ori_img, mesh_img])

                #         img = np.ascontiguousarray(ori_img)
                #         for j, (x, y) in enumerate(outputs['pred_j2ds'][0][0].detach().cpu().numpy()):
                #             cv2.circle(img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)
                #             cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                #                         1, (0,255,255), 1, cv2.LINE_AA)

                #         for j, (x, y) in enumerate(targets[0]['j2ds'][0].cpu().numpy()):
                #             cv2.circle(img, (int(round(x)), int(round(y))), 3, (255,0,0), -1, cv2.LINE_AA)
                #             cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                #                         1, (255,0,0), 1, cv2.LINE_AA)
                #         cv2.imwrite("kp2d_test_eval.jpg", img)
                #     else:
                #         mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                 verts = selected_verts,
                #                                 smpl_faces = smpl_layer.faces,
                #                                 colors = colors,
                #                                 cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                #         full_img = np.hstack([ori_img, mesh_img])
                #     cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        # distributed gather at batch level
        if distributed:
            batch_count = accelerator.gather_for_metrics(batch_count)
            batch_miss_count = accelerator.gather_for_metrics(batch_miss_count)
            batch_fp = accelerator.gather_for_metrics(batch_fp)
            batch_mve = accelerator.gather_for_metrics(batch_mve)
            batch_pa_mve = accelerator.gather_for_metrics(batch_pa_mve)
            batch_mpjpe = accelerator.gather_for_metrics(batch_mpjpe)
            batch_pa_mpjpe = accelerator.gather_for_metrics(batch_pa_mpjpe)

        # update totals
        total_count += sum(batch_count)
        total_miss_count += sum(batch_miss_count)
        total_fp += sum(batch_fp)
        mve += batch_mve
        pa_mve += batch_pa_mve
        mpjpe += batch_mpjpe
        pa_mpjpe += batch_pa_mpjpe

        progress_bar.update(1)

    # collect all results and drop placeholder inf
    mve = np.concatenate([item[1:] for item in mve], axis=0).tolist() if len(mve) > 0 else []
    pa_mve = np.concatenate([item[1:] for item in pa_mve], axis=0).tolist() if len(pa_mve) > 0 else []
    mpjpe = np.concatenate([item[1:] for item in mpjpe], axis=0).tolist() if len(mpjpe) > 0 else []
    pa_mpjpe = np.concatenate([item[1:] for item in pa_mpjpe], axis=0).tolist() if len(pa_mpjpe) > 0 else []

    if len(mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe)/len(mpjpe)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe)/len(pa_mpjpe)), 1)
    error_dict['MVE'] = round(float(sum(mve)/len(mve)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve)/len(pa_mve)), 1)
    error_dict['NUM'] = len(infer_dataloader)

    with open('error_dict.json', 'w') as f:
        json.dump(error_dict, f, indent=2)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict

def shrink_and_expand(bboxes_cxcywh: torch.Tensor, shrink_from=0.1, expand_to=0.0, top_expand=0.05):
    cx, cy, w, h = bboxes_cxcywh.unbind(-1)

    w_orig = w / (1 + 2 * shrink_from)
    h_orig = h / (1 + 2 * shrink_from)

    w_new = w_orig * (1 + 2 * expand_to)
    h_new = h_orig * (1 + 2 * expand_to)

    cy = cy - (h_new * top_expand / 2.0)
    h_new = h_new * (1 + top_expand)

    return torch.stack([cx, cy, w_new, h_new], dim=-1)


# def shrink_and_expand(bboxes_cxcywh: torch.Tensor, shrink_from=0.05, expand_to=0.0, top_expand=0.05):
#     cx, cy, w, h = bboxes_cxcywh.unbind(-1)

#     w_orig = w / (1 + 2 * shrink_from)
#     h_orig = h / (1 + 2 * shrink_from)

#     w_new = w_orig * (1 + 2 * expand_to)
#     h_new = h_orig * (1 + 2 * expand_to)

#     return torch.stack([cx, cy, w_new, h_new], dim=-1)


def eval_bedlam_track(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None, args=None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    if hasattr(model, 'human_model'):
        smpl_layer = model.human_model
    else:
        smpl_layer = model.smplx
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    # yolo = YOLO("data/pretrain/yolov8x.pt")
    # detector = YOLOv8('weights/yolov8m.onnx', conf_thres=0.1, iou_thres=0.5)

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')

    save_path = os.path.join(results_save_path, 'pred.json')
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

    shot_detector = ContentDetector(threshold=50.0, min_scene_len=3)
    frame_count = 0
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    # store arrays per sample then concatenate once at the end
    mve, mpjpe, pa_mpjpe, pa_mve = [], [], [], []

    model.query_bank = []
    if hasattr(model, 'module'):
        model.module.query_bank = []
        
    detect_conf = 0.3
    for itr, (samples, targets) in enumerate(infer_dataloader):
        model.tracker.frame_id = itr
        model.img_seq_idx = itr
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
            outputs = model(samples, targets)

        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            if itr > -1:
                # # # #pred
                if model.__class__.__name__ == "PHMR":
                    pred_verts = outputs[idx]['vertices'].detach().cpu().numpy()
                elif model.__class__.__name__ == "Sam2Model":
                    pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
                elif model.__class__.__name__ == "VideoModel":
                    pred_verts = outputs['pred_verts'].detach().cpu().numpy()
                else:
                    select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
                    pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()
                
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img[img_size[0]:,:,:] = 255
                ori_img[:,img_size[1]:,:] = 255
                ori_img[img_size[0]:,img_size[1]:,:] = 255

                if model.__class__.__name__ == "PHMR":
                    ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                    colors = get_colors_rgb(len(pred_verts))
                    # pred_verts = smplx_out.vertices
                    # smpl_layer = smpl
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                                verts = pred_verts,
                                                smpl_faces = smpl_layer.faces,
                                                cam_intrinsics = batch[idx]['cam_int_original'][0].reshape(3,3).detach().cpu(),
                                                colors=colors)[:img_size[0],:img_size[1]]
                    cv2.imwrite(os.path.join(results_save_path, f'tt_{img_name}.png'), pred_mesh_img)
                elif model.__class__.__name__ == "Sam2Model":
                    ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
                    colors = get_colors_rgb(len(pred_verts))
                    # pred_verts = smplx_out.vertices
                    # smpl_layer = smpl
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                                verts = pred_verts,
                                                smpl_faces = smpl_layer.faces,
                                                cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                                colors=colors)[:img_size[0],:img_size[1]]
                    cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
                else:
                    ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
                    # sat_img = vis_sat(ori_img.copy(),
                    #                     input_size = model.input_size,
                    #                     patch_size = 14,
                    #                     sat_dict = outputs['sat'],
                    #                     bid = idx)[:img_size[0],:img_size[1]]

                    if model.__class__.__name__ == "VideoModel":
                        colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
                    else:
                        colors = get_colors_rgb(len(pred_verts))
                    # colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
                    color_ref = init_color_ref(2000)
                    colors = color_ref[outputs['track_ids'][0]]
                    # h, w, _ = ori_img.shape
                    # ori_img = np.zeros((h, w, 3), dtype=np.uint8)  # RGBA
                    # colors = get_colors_rgb(len(pred_verts))
                    pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                                verts = pred_verts[idx],
                                                # verts = pred_verts,
                                                smpl_faces = smpl_layer.faces,
                                                cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                                colors=colors)[:img_size[0],:img_size[1]]
                        
                    # if 'enc_outputs' not in outputs:
                    #     pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
                    # else:
                    #     enc_out = outputs['enc_outputs']
                    #     h, w = enc_out['hw'][idx]
                    #     flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    #     ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    #     xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    #     scale_map = torch.zeros((h,w,2))
                    #     scale_map[ys,xs] = flatten_map

                    #     pred_scale_img = vis_scale_img(img = ori_img.copy(),
                    #                                     scale_map = scale_map,
                    #                                     conf_thresh = model.sat_cfg['conf_thresh'],
                    #                                     patch_size=28)[:img_size[0],:img_size[1]]

                    # pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                    # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                    # pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                    # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                    #                                                                             np.hstack([pred_scale_img, sat_img])]))

                    # pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
                    # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
                    # pred_boxes = outputs['pred_boxes'][idx].detach().cpu() * model.input_size
                    # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

                    # for box in pred_boxes.int().tolist():
                    #     x1, y1, x2, y2 = box
                    #     cv2.rectangle(pred_mesh_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # for box_idx, tid in enumerate(outputs['active_ids']):
                    #     x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
                    #     # kfx1, kfy1, kfx2, kfy2 = model.tracker.tracks[tid]['kf'].get_xyxy().tolist()
                    #     cv2.rectangle(pred_mesh_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    #     # cv2.rectangle(pred_mesh_img, (int(kfx1), int(kfy1)), (int(kfx2), int(kfy2)), (0, 255, 0), 4)
                    #     cv2.putText(
                    #         pred_mesh_img,
                    #         f"{tid}",
                    #         (x1, y1 - 5),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.8,
                    #         (255, 0, 255),
                    #         2,
                    #         cv2.LINE_AA
                    #     )
                    # pred_mesh_img = vis_boxes(pred_mesh_img, pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                    # select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
                    # ori_pred_boxes = outputs['ori_pred_boxes'][idx][select_queries_idx]
                    # ori_pred_boxes = box_cxcywh_to_xyxy(ori_pred_boxes) * model.input_size
                    # pred_mesh_img = vis_boxes(pred_mesh_img, ori_pred_boxes, color = (0,255,0))[:img_size[0],:img_size[1]]
                    cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), pred_mesh_img)
                    # import pdb; pdb.set_trace()

            pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
            pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
            for box_idx, tid in enumerate(outputs['active_ids']):
                x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
                print(itr, tid, x1, y1, x2, y2)

            # print(len(outputs['pred_verts'][0]))
            # print(model.tracker.active_ids)
            ori_img_size = targets[0]['ori_img_size']
            img_size = targets[0]['img_size']
            def save_tracking_to_csv(frame_id, tracker, file_path, img_size, ori_img_size, overlap_iou=0.8):
                selected_tids = set(tracker.select_tracks_by_self_iou(overlap_iou))
                # if frame_id >= 117:
                # print(tracker.active_ids)
                if not selected_tids:
                    return
                    
                
                # _pred_boxes = shrink_and_expand(outputs['pred_boxes'][0]).detach().cpu() * model.input_size
                _pred_boxes = shrink_and_expand(outputs['pred_j2d_boxes'][0]).detach().cpu() 
                # _pred_boxes = j2ds_to_bboxes_xywh(outputs['pred_j2ds'][-1], 0.0)
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for box_idx, tid in enumerate(tracker.active_ids):
                        if tid not in selected_tids:
                            continue
                        # tinfo = tracker.tracks[tid]
                        # x1, y1, x2, y2 = np.array(tinfo["box"].tolist())

                        pred_boxes = _pred_boxes[box_idx]
                        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
                        x1, y1, x2, y2 = pred_boxes.int().tolist()
                        # print(x1,y1,x2,y2)
                        # x1 = int(x1 * ori_img_size[0] / img_size[0] )
                        # x2 = int(x2 * ori_img_size[0] / img_size[0] )
                        # y1 = int(y1 * ori_img_size[1] / img_size[1] )
                        # y2 = int(y2 * ori_img_size[1] / img_size[1] )
                        # x1 = int(x1 * ori_img_size[0] / self.model.input_size )
                        # x2 = int(x2 * ori_img_size[0] / self.model.input_size )
                        # y1 = int(y1 * ori_img_size[1] / self.model.input_size )
                        # y2 = int(y2 * ori_img_size[1] / self.model.input_size )
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        writer.writerow([frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0])
                        # writer.writerow([frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0])
            # def save_tracking_to_csv(frame_id, tracker, file_path=save_path):
            #     with open(file_path, "a", newline="") as f:
            #         writer = csv.writer(f)
            #         for tid, tinfo in tracker.tracks.items():
            #             x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) 
            #             # x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) / model.input_size
            #             x1, x2 = int(x1 * ori_img_size[0] / img_size[0]), int(x2 * ori_img_size[0] / img_size[0])
            #             y1, y2 = int(y1 * ori_img_size[1] / img_size[1]), int(y2 * ori_img_size[1] / img_size[1])
            #             row = [frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0]
            #             writer.writerow(row)
            # save_tracking_to_csv(itr+1, model.tracker)
            # import pdb; pdb.set_trace()
            save_tracking_to_csv(
                frame_id=itr+1,
                tracker=model.tracker,
                file_path=save_path,
                img_size=img_size,
                ori_img_size=ori_img_size,
                overlap_iou=0.8
            )
        
            # per-batch aggregators
            batch_count = []
            batch_miss_count = []
            batch_fp = []
            batch_mve = []
            batch_mpjpe = []
            batch_pa_mve = []
            batch_pa_mpjpe = []

            bs = len(targets)
            for idx in range(bs):
                batch_count.append(0)
                batch_miss_count.append(0)
                batch_fp.append(0)
                # avoid empty gather
                sample_mve = [float('inf')]
                sample_pa_mve = [float('inf')]
                sample_mpjpe = [float('inf')]
                sample_pa_mpjpe = [float('inf')]

                # gt 
                gt_verts = targets[idx]['verts'].to(device=cur_device)
                gt_transl = targets[idx]['transl'].to(device=cur_device)
                gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:,None,:], smpl2h36m_regressor]) + gt_transl[:,None,:]

                gt_verts = gt_verts.cpu().numpy()
                gt_j3ds = gt_j3ds.cpu().numpy()
                gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]

                select_queries_idx = torch.where(outputs['pred_confs'][idx] > model.tracker.conf_thresh)[0]
                pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
                pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
                pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]
                
                pred_verts = pred_verts.cpu().numpy()
                pred_j3ds = pred_j3ds.cpu().numpy()
                pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]

                # matching
                matched_verts_idx = []
                assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
                greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds)  # tuples are (idx_pred_kps, idx_gt_kps)
                matchDict, falsePositive_count = get_matching_dict(greedy_match)

                if model.__class__.__name__ == "PHMR" or model.__class__.__name__ == "Sam2Model":
                    gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
                    gtIdxs = np.arange(len(gt_j3ds))
                    for gtIdx in gtIdxs:
                        gt_verts_list.append(gt_verts[gtIdx])
                        gt_joints_list.append(gt_j3ds[gtIdx])
                        pred_joints_list.append(pred_j3ds[gtIdx])
                        pred_verts_list.append(pred_verts[gtIdx])
                        matched_verts_idx.append(gtIdx)
                else:
                    # align with matching result
                    gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
                    gtIdxs = np.arange(len(gt_j3ds))
                    miss_flag = []
                    for gtIdx in gtIdxs:
                        gt_verts_list.append(gt_verts[gtIdx])
                        gt_joints_list.append(gt_j3ds[gtIdx])
                        if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                            miss_flag.append(1)
                            pred_verts_list.append([])
                            pred_joints_list.append([])
                        else:
                            miss_flag.append(0)
                            pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                            pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                            matched_verts_idx.append(matchDict[str(gtIdx)])

                # compute 3D errors
                for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                    batch_count[-1] += 1

                    if model.__class__.__name__ != "PHMR" and model.__class__.__name__ != "Sam2Model":
                        if miss_flag[i] == 1:
                            batch_miss_count[-1] += 1
                            continue

                    gt3d = gt3d.reshape(-1, 3)
                    pred3d = pred.reshape(-1, 3)
                    gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                    pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                    gt_pelvis = gt3d[[0],:].copy()
                    pred_pelvis = pred3d[[0],:].copy()

                    gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                    gt3d_verts = (gt3d_verts - gt_pelvis).copy()
                    
                    pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                    pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                    # joints
                    # reconstruction_error
                    error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                    sample_mpjpe.append(float(error_j))
                    sample_pa_mpjpe.append(float(pa_error_j))
                    # vertices
                    error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                    sample_mve.append(float(error_v))
                    sample_pa_mve.append(float(pa_error_v))

                # counting and visualization
                step += 1
                batch_fp[-1] += falsePositive_count

                # stash per-sample arrays (with leading inf)
                batch_mve.append(np.array(sample_mve))
                batch_pa_mve.append(np.array(sample_pa_mve))
                batch_mpjpe.append(np.array(sample_mpjpe))
                batch_pa_mpjpe.append(np.array(sample_pa_mpjpe))
                # print(sample_mve, sample_pa_mve, sample_mpjpe, sample_pa_mpjpe)

                # img_idx = step + accelerator.process_index*len(infer_dataloader)*bs
                # if vis and (img_idx%vis_step == 0) and len(matched_verts_idx) > 0:
                #     img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                #     ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                #     ori_img = pad_img(ori_img, model.input_size)

                #     print(sample_mve, sample_pa_mve, sample_mpjpe, sample_pa_mpjpe)

                #     selected_verts = pred_verts[matched_verts_idx]
                #     colors = get_colors_rgb(len(selected_verts))
                #     if model.__class__.__name__ == "PHMR":
                #         mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                     verts = selected_verts,
                #                                     smpl_faces = smpl_layer.faces,
                #                                     colors = colors,
                #                                     cam_intrinsics = outputs[idx]['cam_int_original'][matched_verts_idx].detach().cpu())
                #         full_img = np.hstack([ori_img, mesh_img])
                #     elif model.__class__.__name__ == "Sam2Model":
                #         mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                     verts = selected_verts,
                #                                     smpl_faces = smpl_layer.faces,
                #                                     colors = colors,
                #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                #         full_img = np.hstack([ori_img, mesh_img])

                #         img = np.ascontiguousarray(ori_img)
                #         for j, (x, y) in enumerate(outputs['pred_j2ds'][0][0].detach().cpu().numpy()):
                #             cv2.circle(img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)
                #             cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                #                         1, (0,255,255), 1, cv2.LINE_AA)

                #         for j, (x, y) in enumerate(targets[0]['j2ds'][0].cpu().numpy()):
                #             cv2.circle(img, (int(round(x)), int(round(y))), 3, (255,0,0), -1, cv2.LINE_AA)
                #             cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                #                         1, (255,0,0), 1, cv2.LINE_AA)
                #         cv2.imwrite("kp2d_test_eval.jpg", img)
                #     else:
                #         mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                 verts = selected_verts,
                #                                 smpl_faces = smpl_layer.faces,
                #                                 colors = colors,
                #                                 cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                #         full_img = np.hstack([ori_img, mesh_img])
                #     cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        # distributed gather at batch level
        if distributed:
            batch_count = accelerator.gather_for_metrics(batch_count)
            batch_miss_count = accelerator.gather_for_metrics(batch_miss_count)
            batch_fp = accelerator.gather_for_metrics(batch_fp)
            batch_mve = accelerator.gather_for_metrics(batch_mve)
            batch_pa_mve = accelerator.gather_for_metrics(batch_pa_mve)
            batch_mpjpe = accelerator.gather_for_metrics(batch_mpjpe)
            batch_pa_mpjpe = accelerator.gather_for_metrics(batch_pa_mpjpe)

        # update totals
        total_count += sum(batch_count)
        total_miss_count += sum(batch_miss_count)
        total_fp += sum(batch_fp)
        mve += batch_mve
        pa_mve += batch_pa_mve
        mpjpe += batch_mpjpe
        pa_mpjpe += batch_pa_mpjpe

        progress_bar.update(1)

    # collect all results and drop placeholder inf
    mve = np.concatenate([item[1:] for item in mve], axis=0).tolist() if len(mve) > 0 else []
    pa_mve = np.concatenate([item[1:] for item in pa_mve], axis=0).tolist() if len(pa_mve) > 0 else []
    mpjpe = np.concatenate([item[1:] for item in mpjpe], axis=0).tolist() if len(mpjpe) > 0 else []
    pa_mpjpe = np.concatenate([item[1:] for item in pa_mpjpe], axis=0).tolist() if len(pa_mpjpe) > 0 else []

    if len(mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe)/len(mpjpe)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe)/len(pa_mpjpe)), 1)
    error_dict['MVE'] = round(float(sum(mve)/len(mve)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve)/len(pa_mve)), 1)
    error_dict['NUM'] = len(infer_dataloader)

    with open('error_dict.json', 'w') as f:
        json.dump(error_dict, f, indent=2)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict
    

def eval_posetrack_video(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None, args=None, sam2_tracker=None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    if hasattr(model, 'human_model'):
        smpl_layer = model.human_model
    else:
        smpl_layer = model.smplx
    # yolo = YOLO("data/pretrain/yolov8x.pt")
    # detector = YOLOv8('weights/yolov8m.onnx', conf_thres=0.1, iou_thres=0.5)

    video_dir = infer_dataloader.dataset.dataset_path

    # frame_names = [
    #     p for p in os.listdir(video_dir)
    #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    # ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = sam2_tracker.init_state(video_path=video_dir)
    sam2_tracker.reset_state(inference_state)

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')

    save_path = os.path.join(results_save_path, 'pred.json')
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

    first_samples, first_targets = next(iter(infer_dataloader))
    def to_device(batch, device):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            t = [to_device(x, device) for x in batch]
            return type(batch)(t) if isinstance(batch, tuple) else t
        return batch 

    first_samples = to_device(first_samples, cur_device)
    first_targets = to_device(first_targets, cur_device)

    outputs = model.forward_first_batch(first_samples, first_targets)
    select_queries_idx = torch.where(outputs[0]['pred_confs'][0] > model.tracker.conf_thresh)[0]
    body_bboxes = outputs[0]['pred_boxes'][0][select_queries_idx].detach().cpu().numpy() * model.input_size


    ann_obj_id = np.array([i+1 for i in range(body_bboxes.shape[0])])
    ann_frame_idx = 0
    import pdb; pdb.set_trace()
    for box, obj_idx in zip(body_bboxes, ann_obj_id):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_idx,
            box=box,
        )

    
    detect_conf = 0.3
    for itr, (samples, targets, (out_frame_idx, out_obj_ids, out_mask_logits)) in enumerate(zip(infer_dataloader, predictor.propagate_in_video(inference_state))):
        if model.__class__.__name__ == "Sam2Model":
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(samples[0].device)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(samples[0].device)
            img = samples[0] * std + mean
            img = torch.clamp(img, 0, 255).to(torch.uint8)
    
            detected = detector(img.cpu().numpy().transpose(1,2,0))
            boxes = detected[0][detected[2] == 0]
            targets[0]['boxes'] = box_xyxy_to_cxcywh(torch.tensor(boxes)) / model.input_size 
            targets[0]['cam_intrinsics'] = model.cam_intrinsics[None]

            with torch.no_grad():    
                outputs = model(samples, targets)
        else:
            samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
            with torch.no_grad():    
                outputs = model(samples, targets)

        # bs = len(targets)
        # for idx in range(bs):
        #     img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
        #     img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

        #     #pred
        #     if model.__class__.__name__ == "PHMR":
        #         pred_verts = outputs[idx]['vertices'].detach().cpu().numpy()
        #     elif model.__class__.__name__ == "Sam2Model":
        #         pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
        #     elif model.__class__.__name__ == "VideoModel":
        #         pred_verts = outputs['pred_verts'].detach().cpu().numpy()
        #     else:
        #         select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
        #         pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

        #     ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
        #     ori_img[img_size[0]:,:,:] = 255
        #     ori_img[:,img_size[1]:,:] = 255
        #     ori_img[img_size[0]:,img_size[1]:,:] = 255

        #     if model.__class__.__name__ == "PHMR":
        #         ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
        #         colors = get_colors_rgb(len(pred_verts))
        #         # pred_verts = smplx_out.vertices
        #         # smpl_layer = smpl
        #         pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
        #                                     verts = pred_verts,
        #                                     smpl_faces = smpl_layer.faces,
        #                                     cam_intrinsics = batch[idx]['cam_int_original'][0].reshape(3,3).detach().cpu(),
        #                                     colors=colors)[:img_size[0],:img_size[1]]
        #         cv2.imwrite(os.path.join(results_save_path, f'tt_{img_name}.png'), pred_mesh_img)
        #     elif model.__class__.__name__ == "Sam2Model":
        #         ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
        #         colors = get_colors_rgb(len(pred_verts))
        #         # pred_verts = smplx_out.vertices
        #         # smpl_layer = smpl
        #         pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
        #                                     verts = pred_verts,
        #                                     smpl_faces = smpl_layer.faces,
        #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
        #                                     colors=colors)[:img_size[0],:img_size[1]]
        #         cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
        #     else:
        #         ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
        #         # sat_img = vis_sat(ori_img.copy(),
        #         #                     input_size = model.input_size,
        #         #                     patch_size = 14,
        #         #                     sat_dict = outputs['sat'],
        #         #                     bid = idx)[:img_size[0],:img_size[1]]

        #         if model.__class__.__name__ == "VideoModel":
        #             colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
        #         else:
        #             colors = get_colors_rgb(len(pred_verts))
        #         pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
        #                                     verts = pred_verts[idx],
        #                                     smpl_faces = smpl_layer.faces,
        #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
        #                                     colors=colors)[:img_size[0],:img_size[1]]
                    
        #         # if 'enc_outputs' not in outputs:
        #         #     pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
        #         # else:
        #         #     enc_out = outputs['enc_outputs']
        #         #     h, w = enc_out['hw'][idx]
        #         #     flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

        #         #     ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
        #         #     xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
        #         #     scale_map = torch.zeros((h,w,2))
        #         #     scale_map[ys,xs] = flatten_map

        #         #     pred_scale_img = vis_scale_img(img = ori_img.copy(),
        #         #                                     scale_map = scale_map,
        #         #                                     conf_thresh = model.sat_cfg['conf_thresh'],
        #         #                                     patch_size=28)[:img_size[0],:img_size[1]]

        #         # pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
        #         # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
        #         # pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

        #         # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
        #         #                                                                             np.hstack([pred_scale_img, sat_img])]))

        #         pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
        #         pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
        #         # pred_boxes = outputs['pred_boxes'][idx].detach().cpu() * model.input_size
        #         # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

        #         for box_idx, tid in enumerate(outputs['active_ids']):
        #             x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
        #             cv2.rectangle(pred_mesh_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #             cv2.putText(
        #                 pred_mesh_img,
        #                 f"ID {tid}",
        #                 (x1, y1 - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.8,
        #                 (255, 0, 255),
        #                 2,
        #                 cv2.LINE_AA
        #             )
        #         pred_mesh_img = vis_boxes(pred_mesh_img, pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

        #         # select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
        #         # ori_pred_boxes = outputs['ori_pred_boxes'][idx][select_queries_idx]
        #         # ori_pred_boxes = box_cxcywh_to_xyxy(ori_pred_boxes) * model.input_size
        #         # pred_mesh_img = vis_boxes(pred_mesh_img, ori_pred_boxes, color = (0,255,0))[:img_size[0],:img_size[1]]
        #         cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), pred_mesh_img)

            # print(len(outputs['pred_verts'][0]))
            # print(model.tracker.active_ids)
            ori_img_size = targets[0]['ori_img_size']
            img_size = targets[0]['img_size']
            def save_tracking_to_csv(frame_id, tracker, file_path, img_size, ori_img_size, overlap_iou=0.8):
                selected_tids = set(tracker.select_tracks_by_self_iou(overlap_iou))
                if not selected_tids:
                    return
                # print(frame_id, tracker.active_ids)
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for tid in tracker.active_ids:
                        if tid not in selected_tids:
                            continue
                        tinfo = tracker.tracks[tid]
                        x1, y1, x2, y2 = np.array(tinfo["box"].tolist())
                        x1 = int(x1 * ori_img_size[0] / img_size[0])
                        x2 = int(x2 * ori_img_size[0] / img_size[0])
                        y1 = int(y1 * ori_img_size[1] / img_size[1])
                        y2 = int(y2 * ori_img_size[1] / img_size[1])
                        writer.writerow([frame_id, tid, x1, y1, x2-x1, y2-y1, 1, 0, 0, 0])
            # def save_tracking_to_csv(frame_id, tracker, file_path=save_path):
            #     with open(file_path, "a", newline="") as f:
            #         writer = csv.writer(f)
            #         for tid, tinfo in tracker.tracks.items():
            #             x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) 
            #             # x1, y1, x2, y2 = np.array(tinfo["box"].tolist()) / model.input_size
            #             x1, x2 = int(x1 * ori_img_size[0] / img_size[0]), int(x2 * ori_img_size[0] / img_size[0])
            #             y1, y2 = int(y1 * ori_img_size[1] / img_size[1]), int(y2 * ori_img_size[1] / img_size[1])
            #             row = [frame_id, tid, x1, y1, x2, y2, 1, 0, 0, 0]
            #             writer.writerow(row)
            # save_tracking_to_csv(itr+1, model.tracker)
            # import pdb; pdb.set_trace()
            save_tracking_to_csv(
                frame_id=itr+1,
                tracker=model.tracker,
                file_path=save_path,
                img_size=img_size,
                ori_img_size=ori_img_size,
                overlap_iou=0.8
            )



        progress_bar.update(1)