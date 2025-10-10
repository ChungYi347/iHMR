import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.process_batch import prepare_batch
import time
import cv2
import trimesh

# from ultralytics import YOLO
import sys
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ONNX-YOLOv8-Object-Detection"))
from yolov8 import YOLOv8
from models.geometry import axis_angle_to_matrix

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
                                            cam_intrinsics = outputs['pred_intrinsics'][idx][0].reshape(3,3).detach().cpu(),
                                            colors=colors)[:img_size[0],:img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
            else:
                ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)[:img_size[0],:img_size[1]]

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

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                                                                                            np.hstack([pred_scale_img, sat_img])]))


        progress_bar.update(1)
    

