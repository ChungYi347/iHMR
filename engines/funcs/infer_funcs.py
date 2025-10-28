import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb, get_colors_rgb_tracking_ids
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
import csv
from scenedetect.detectors import ContentDetector

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

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

                cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
                                                                                            np.hstack([pred_scale_img, sat_img])]))


        progress_bar.update(1)
    

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
    
    detect_conf = 0.3
    for itr, (samples, targets) in enumerate(infer_dataloader):
        model.tracker.frame_id = itr
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
            img = tensor_to_BGR(unNormalize(samples[0].cpu()))[:,:,::-1]
            shots = shot_detector.process_frame(frame_count, img)
            frame_count += 1
            is_new_shot = len(shots) > 0 if frame_count > 1 else False
            # print(targets[0]['img_path'], shots, is_new_shot)
            # model.tracker.reset()
            if is_new_shot:
                model.tracker.reset()
                print(model.tracker.next_id)
                print(targets[0]['img_path'], shots)
            
            samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
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

            # # # # #pred
            # if model.__class__.__name__ == "PHMR":
            #     pred_verts = outputs[idx]['vertices'].detach().cpu().numpy()
            # elif model.__class__.__name__ == "Sam2Model":
            #     pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
            # elif model.__class__.__name__ == "VideoModel":
            #     pred_verts = outputs['pred_verts'].detach().cpu().numpy()
            # else:
            #     select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            #     pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()
            
            # ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            # ori_img[img_size[0]:,:,:] = 255
            # ori_img[:,img_size[1]:,:] = 255
            # ori_img[img_size[0]:,img_size[1]:,:] = 255

            # if model.__class__.__name__ == "PHMR":
            #     ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
            #     colors = get_colors_rgb(len(pred_verts))
            #     # pred_verts = smplx_out.vertices
            #     # smpl_layer = smpl
            #     pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
            #                                 verts = pred_verts,
            #                                 smpl_faces = smpl_layer.faces,
            #                                 cam_intrinsics = batch[idx]['cam_int_original'][0].reshape(3,3).detach().cpu(),
            #                                 colors=colors)[:img_size[0],:img_size[1]]
            #     cv2.imwrite(os.path.join(results_save_path, f'tt_{img_name}.png'), pred_mesh_img)
            # elif model.__class__.__name__ == "Sam2Model":
            #     ori_img = pad_img(ori_img, max(img_size), pad_color_offset=255)
            #     colors = get_colors_rgb(len(pred_verts))
            #     # pred_verts = smplx_out.vertices
            #     # smpl_layer = smpl
            #     pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
            #                                 verts = pred_verts,
            #                                 smpl_faces = smpl_layer.faces,
            #                                 cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
            #                                 colors=colors)[:img_size[0],:img_size[1]]
            #     cv2.imwrite(os.path.join(results_save_path, f'sam2_{img_name}.png'), pred_mesh_img)
            # else:
            #     ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)
            #     # sat_img = vis_sat(ori_img.copy(),
            #     #                     input_size = model.input_size,
            #     #                     patch_size = 14,
            #     #                     sat_dict = outputs['sat'],
            #     #                     bid = idx)[:img_size[0],:img_size[1]]

            #     if model.__class__.__name__ == "VideoModel":
            #         colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
            #     else:
            #         colors = get_colors_rgb(len(pred_verts))
            #     colors = get_colors_rgb_tracking_ids(outputs['track_ids'][0])
            #     colors = get_colors_rgb(len(pred_verts))
            #     pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
            #                                 # verts = pred_verts[idx],
            #                                 verts = pred_verts,
            #                                 smpl_faces = smpl_layer.faces,
            #                                 cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
            #                                 colors=colors)[:img_size[0],:img_size[1]]
                    
            #     # if 'enc_outputs' not in outputs:
            #     #     pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
            #     # else:
            #     #     enc_out = outputs['enc_outputs']
            #     #     h, w = enc_out['hw'][idx]
            #     #     flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

            #     #     ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
            #     #     xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
            #     #     scale_map = torch.zeros((h,w,2))
            #     #     scale_map[ys,xs] = flatten_map

            #     #     pred_scale_img = vis_scale_img(img = ori_img.copy(),
            #     #                                     scale_map = scale_map,
            #     #                                     conf_thresh = model.sat_cfg['conf_thresh'],
            #     #                                     patch_size=28)[:img_size[0],:img_size[1]]

            #     # pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
            #     # pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
            #     # pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

            #     # cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), np.vstack([np.hstack([pred_box_img, pred_mesh_img]),
            #     #                                                                             np.hstack([pred_scale_img, sat_img])]))

            #     pred_boxes = outputs['pred_j2d_boxes'][idx].detach().cpu()
            #     pred_boxes = box_cxcywh_to_xyxy(pred_boxes) 
            #     # pred_boxes = outputs['pred_boxes'][idx].detach().cpu() * model.input_size
            #     # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

            #     for box_idx, tid in enumerate(outputs['active_ids']):
            #         x1, y1, x2, y2 = pred_boxes[box_idx].int().tolist()
            #         # kfx1, kfy1, kfx2, kfy2 = model.tracker.tracks[tid]['kf'].get_xyxy().tolist()
            #         cv2.rectangle(pred_mesh_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            #         # cv2.rectangle(pred_mesh_img, (int(kfx1), int(kfy1)), (int(kfx2), int(kfy2)), (0, 255, 0), 4)
            #         cv2.putText(
            #             pred_mesh_img,
            #             f"{tid}",
            #             (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.8,
            #             (255, 0, 255),
            #             2,
            #             cv2.LINE_AA
            #         )
            #     pred_mesh_img = vis_boxes(pred_mesh_img, pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

            #     # select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            #     # ori_pred_boxes = outputs['ori_pred_boxes'][idx][select_queries_idx]
            #     # ori_pred_boxes = box_cxcywh_to_xyxy(ori_pred_boxes) * model.input_size
            #     # pred_mesh_img = vis_boxes(pred_mesh_img, ori_pred_boxes, color = (0,255,0))[:img_size[0],:img_size[1]]
            #     cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'), pred_mesh_img)
            #     import pdb; pdb.set_trace()

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



        progress_bar.update(1)
    

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
    select_queries_idx = torch.where(outputs[0]['pred_confs'][0] > conf_thresh)[0]
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