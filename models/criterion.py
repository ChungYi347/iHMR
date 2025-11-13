# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)
import os
import torch
from torch import nn
import torch.nn.functional as F
from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from collections import defaultdict

def focal_loss(inputs, targets, valid_mask = None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # prob = inputs.sigmoid()
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # if valid_mask is not None:
    #     loss = loss * valid_mask
    
    return loss.mean()

class SetCriterionHMR(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  weight_dict, losses = ['poses','betas', 'j3ds','j2ds', 'depths'], 
                focal_alpha=0.25, focal_gamma = 2.0, j2ds_norm_scale = 518):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        self.weight_dict = weight_dict
        

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, num_instances, **kwargs):
        total_loss = 0.0
        denom = 0.0  # num_instances가 없으면 여기서 계산

        device = self.device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device) / num_instances

        losses = {}
        loss_mask = None

        # output_key = 'pred_'+loss
        # if loss == 'j3ds' or loss == 'j2ds':
        #     output_pnums = [out[0].shape[0] for out in outputs['pred_'+loss]]
        # else:
        # output_pnums = [out.shape[0] for out in outputs['pred_'+loss]]
        # target_pnums = [t['pnum'] for t in targets]

        # print(output_pnums, target_pnums)
        src = [ outputs[loss][b][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]]

        if len(src) == 0:
            losses[loss] = total_loss.squeeze(0)
            return losses

        src = torch.cat(src, dim=0)
        if loss == 'j2ds':
            target = torch.cat([targets[b]['padd_'+loss][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]], dim=0)
        else:
            target = torch.cat([targets[b][loss][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]], dim=0)

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([
                targets[b]['j2ds_mask'][:targets[b]['pnum'], :24, :]
                for b in range(len(targets)) if targets[b]['pnum'] > 0
            ], dim=0)
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
            loss_mask = loss_mask[:,:24,:]
        elif loss == 'poses':
            src = src.view(src.shape[0], -1)
            target = target[:,:src.shape[1]]

        # if loss == 'j2ds':
        #     from utils.transforms import unNormalize
        #     from utils.visualization import tensor_to_BGR, pad_img
        #     import numpy as np
        #     import cv2
        #     img = np.ascontiguousarray(tensor_to_BGR(unNormalize(outputs['img'][0]).cpu()))
        #     for j, (x, y) in enumerate(src[0].detach().cpu().numpy() * self.j2ds_norm_scale):
        #         cv2.circle(img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)
        #         cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (0,255,255), 1, cv2.LINE_AA)

        #     for j, (x, y) in enumerate(target[0].cpu().numpy()  * self.j2ds_norm_scale):
        #         cv2.circle(img, (int(round(x)), int(round(y))), 3, (255,0,0), -1, cv2.LINE_AA)
        #         cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (255,0,0), 1, cv2.LINE_AA)
        #     cv2.imwrite("kp2d.png", img)
        #     import pdb; pdb.set_trace()
        # # elif loss == 'j3ds':
        # #     import pdb; pdb.set_trace()

        valid_loss = torch.abs(src-target)

        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        total_loss = valid_loss.flatten(1).mean(-1).sum()/num_instances
        losses[loss] = total_loss
        return losses

    def loss_L2(self, loss, outputs, targets, num_instances, **kwargs):
        total_loss = 0.0
        denom = 0.0  # num_instances가 없으면 여기서 계산

        device = self.device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device) / num_instances

        losses = {}
        loss_mask = None

        # output_key = loss
        # if loss == 'j3ds' or loss == 'j2ds':
        #     output_pnums = [out[0].shape[0] for out in outputs['pred_'+loss]]
        # else:

        src = [ outputs[ loss][b][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]]

        if len(src) == 0:
            losses[loss] = total_loss.squeeze(0)
            return losses
        
        src = torch.cat(src, dim=0)
        if loss == 'j2ds':
            target = torch.cat([targets[b]['padd_'+loss][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]], dim=0)
        else:
            target = torch.cat([targets[b][loss][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]], dim=0)

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([
                targets[b]['j2ds_mask'][:targets[b]['pnum'], :24 :]
                for b in range(len(targets)) if targets[b]['pnum'] > 0
            ], dim=0)
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
            loss_mask = loss_mask[:,:24,:]
        elif loss == 'poses':
            src = src.view(src.shape[0], -1)
            target = target[:,:src.shape[1]]

            # from utils.transforms import rotation_matrix_to_angle_axis
            # from models.geometry import matrix_to_axis_angle
        
        # valid_loss = torch.abs(src-target)
        valid_loss = F.mse_loss(src, target, reduction='none')

        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        # if loss == 'betas':
        #     valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        total_loss = valid_loss.flatten(1).mean(-1).sum()/num_instances
        losses[loss] = total_loss
        return losses

    def loss_normalized_depths(self, loss, outputs, targets, num_instances, **kwargs):
        assert loss == 'depths' 
        losses = {}

        device = self.device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device) / num_instances

        output_key = loss
        src = [outputs[loss][b][:targets[b]['pnum'], [0]] 
               for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'depths' in targets[b]]

        if len(src) == 0:
            losses[loss] = total_loss.squeeze(0)
            return losses
        src = torch.cat(src, dim=0)

        target  = torch.cat([targets[b]['depths'][:targets[b]['pnum'], [0]] 
               for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'depths' in targets[b]], dim=0)
        target_focals    = torch.cat([targets[b]['focals'][:targets[b]['pnum'], [0]] 
               for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'focals' in targets[b]], dim=0)

        # print(src.shape, target.shape, target_focals.shape)

        src = target_focals * src
        
        assert src.shape == target.shape

        # valid_loss = torch.abs(1./(src + 1e-8) - 1./(target + 1e-8))
        valid_loss = torch.abs(src - 1./(target + 1e-6))
        # valid_loss = F.mse_loss(src, 1./(target + 1e-6), reduction='none')
        total_loss = valid_loss.flatten(1).mean(-1).sum()/num_instances
        losses[loss] = total_loss
        return losses

    def get_loss(self, loss, outputs, targets, num_instances, **kwargs):
        loss_map = {
            # 'confs': self.loss_confs,
            'poses': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            # 'verts': self.loss_L1,
            # 'poses': self.loss_L2,
            # 'betas': self.loss_L2,
            # 'j3ds': self.loss_L2,
            # 'j2ds': self.loss_L2,
            # 'verts': self.loss_L2,
            'depths': self.loss_normalized_depths,
            # 'scale_map': self.loss_scale_map,       
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](loss, outputs, targets, num_instances, **kwargs)


    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            if loss != 'scale_map':
                for t in targets:
                    num_instances += t['pnum'] if loss in t else 0
                num_instances = torch.as_tensor([num_instances], dtype=torch.float32, device=self.device)
            else:
                for t in targets:
                    num_instances += t['scale_map'][...,0].sum().item()
                num_instances = torch.as_tensor([num_instances], dtype=torch.float32, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        return num_valid_instances
        
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # remove invalid information in targets
        for t in targets:
            if not t['3d_valid']:
                for key in ['betas', 'poses', 'j3ds', 'depths', 'focals']:
                    if key in t:
                        del t[key]

        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        self.device = outputs[0]['poses'].device
        num_valid_instances = self.get_valid_instances(targets)

        # Compute all the requested losses
        losses = {k: torch.zeros((), device=self.device, dtype=torch.float32) for k in self.losses}

        outputs = {
            k: torch.stack([d[k] for d in outputs], dim=0)
            if isinstance(outputs[0][k], torch.Tensor)
            else [d[k] for d in outputs]
            for k in outputs[0]
        }
        
        outputs['verts'] = outputs['smpl_vertices']
        for loss_name in self.losses:
            out = self.get_loss(loss_name, outputs, targets, num_valid_instances[loss_name])
            if loss_name in out:
                losses[loss_name] = losses[loss_name] + out[loss_name]

        
        return losses

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses = ['confs','boxes', 'poses','betas', 'j3ds','j2ds', 'depths', 'kid_offsets'], 
                focal_alpha=0.25, focal_gamma = 2.0, j2ds_norm_scale = 518):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        self.weight_dict = weight_dict
        

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None


    def loss_boxes(self, loss, outputs, targets, indices, num_instances, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        assert loss == 'boxes'
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape
        
        src_boxes = src
        target_boxes = target

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['boxes'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['giou'] = loss_giou.sum() / num_instances

        # # calculate the x,y and h,w loss
        # with torch.no_grad():
        #     losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #     losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, indices, num_instances, **kwargs):
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape

        losses = {}
        loss_mask = None

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:45,:]
            target = target[:,:45,:]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([t['j2ds_mask'][i] for t, (_, i) in zip(targets, indices) if 'j2ds' in t], dim=0)
            # Use 45 smpl joints
            src = src[:,:45,:]
            target = target[:,:45,:]
            loss_mask = loss_mask[:,:45,:]

        # if loss == 'j2ds':
        #     if 'img' in outputs:
        #         from utils.transforms import unNormalize
        #         from utils.visualization import tensor_to_BGR, pad_img
        #         import numpy as np
        #         import cv2
        #         img = np.ascontiguousarray(tensor_to_BGR(unNormalize(outputs['img'][0]).cpu()))
        #         for idx in range(len(src)):
        #             for j, (x, y) in enumerate(src[idx].detach().cpu().numpy() * self.j2ds_norm_scale):
        #                 cv2.circle(img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)
        #                 cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                             1, (0,255,255), 1, cv2.LINE_AA)

        #             for j, (x, y) in enumerate(target[idx].cpu().numpy()  * self.j2ds_norm_scale):
        #                 cv2.circle(img, (int(round(x)), int(round(y))), 3, (255,0,0), -1, cv2.LINE_AA)
        #                 cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                             1, (255,0,0), 1, cv2.LINE_AA)
        #         cv2.imwrite("kp2d2.png", img)
        #         img = np.ascontiguousarray(tensor_to_BGR(unNormalize(outputs['img'][0]).cpu()))
        #         cv2.imwrite("kp2d2_img.png", img)
        #         # import pdb; pdb.set_trace()
        # if loss == 'verts':
        #     import numpy as np
        #     from utils.visualization import vis_meshes_img, get_colors_rgb
        #     device = outputs['img'][0]
        #     mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        #     std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        #     img = outputs['img'][0] * std.to(device) + mean.to(device)
        #     img = torch.clamp(img, 0, 255).to(torch.uint8)
        #     img = img.cpu().numpy()
            
        #     img = np.ascontiguousarray(img.transpose(1,2,0))
        #     from models.human_models import SMPL_Layer
        #     from configs.paths import smpl_model_path
        #     import cv2
        #     human_model = SMPL_Layer(model_path = smpl_model_path, with_genders = False)
        #     idx = 0
        #     img_size = img.shape
        #     pred_idx = outputs['pred_confs'][idx] > 0.3
        #     pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()[pred_idx.detach().cpu().numpy()[:, 0]]
        #     colors = get_colors_rgb(len(pred_verts))
        #     pred_mesh_img = vis_meshes_img(img = img.copy(),
        #                                     verts = pred_verts,
        #                                     smpl_faces = human_model.faces,
        #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
        #                                     colors=colors)[:img_size[0],:img_size[1]]
        #     img_name = targets[0]['img_path'].split('/')[-1].split('.')[0]
        #     cv2.imwrite(os.path.join('test_train', f'trn_sam2_{img_name}.png'), pred_mesh_img)
        #     print(os.path.join('test_train', f'trn_sam2_{img_name}.png'))
        #     import pdb; pdb.set_trace()
        
        valid_loss = torch.abs(src-target)

        # if loss == 'j2ds':
        #     print(src.shape)
        #     print(target.shape)
        #     print(num_instances)
        #     exit(0)
        
        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances


        # print(loss, losses[loss])
        return losses

    def loss_scale_map(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'scale_map'

        pred_map = outputs['enc_outputs']['scale_map']
        tgt_map = torch.cat([t['scale_map'] for t in targets], dim=0)
        assert pred_map.shape == tgt_map.shape

        labels = tgt_map[:,0]
        pred_scales = pred_map[:,1]
        tgt_scales = tgt_map[:, 1]

        detection_valid_mask = labels.bool()
        cur = 0
        lens = [len(t['scale_map']) for t in targets]
        for i, tgt in enumerate(targets):
            if tgt['detect_all_people']:
                detection_valid_mask[cur:cur+lens[i]] = True
            cur += lens[i]

     
        losses = {}
        losses['map_confs'] = focal_loss(pred_map[:,0], labels, valid_mask=detection_valid_mask)/1.
        losses['map_scales'] = torch.abs((pred_scales - tgt_scales)[torch.where(labels)[0]]).sum()/num_instances


        return losses

    def loss_confs(self, loss, outputs, targets, indices, num_instances, is_dn=False, **kwargs):
        assert loss == 'confs'
        idx = self._get_src_permutation_idx(indices)
        pred_confs = outputs['pred_'+loss]

        with torch.no_grad():
            labels = torch.zeros_like(pred_confs)
            labels[idx] = 1
            detection_valid_mask = torch.zeros_like(pred_confs,dtype=bool)
            detection_valid_mask[idx] = True
            valid_batch_idx = torch.where(torch.tensor([t['detect_all_people'] for t in targets]))[0]
            detection_valid_mask[valid_batch_idx] = True

        
        losses = {}
        if is_dn:
            losses[loss] = focal_loss(pred_confs, labels) / num_instances
        else:
            losses[loss] = focal_loss(pred_confs, labels, valid_mask = detection_valid_mask) / num_instances

        return losses

    def loss_normalized_depths(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'depths' 
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx][...,[1]]  # [d d/f]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)[...,[0]]
        target_focals = torch.cat([t['focals'][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)

        # print(src.shape, target.shape, target_focals.shape)

        src = target_focals * src
        
        assert src.shape == target.shape

        valid_loss = torch.abs(1./(src + 1e-8) - 1./(target + 1e-8))
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            'confs': self.loss_confs,
            'boxes': self.loss_boxes,
            'poses': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            'verts': self.loss_L1,
            'depths': self.loss_normalized_depths,
            'scale_map': self.loss_scale_map,       
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](loss, outputs, targets, indices, num_instances, **kwargs)


    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            if loss != 'scale_map':
                for t in targets:
                    num_instances += t['pnum'] if loss in t else 0
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            else:
                for t in targets:
                    num_instances += t['scale_map'][...,0].sum().item()
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        return num_valid_instances

    def prep_for_dn(self, dn_meta):
        output_known = dn_meta['output_known']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size//num_dn_groups

        return output_known, single_pad, num_dn_groups

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # remove invalid information in targets
        for t in targets:
            if not t['3d_valid']:
                for key in ['betas', 'poses', 'j3ds', 'depths', 'focals', 'img']:
                    if key in t:
                        del t[key]

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.device = outputs['pred_poses'].device
        num_valid_instances = self.get_valid_instances(targets)

        # Compute all the requested losses
        losses = {}
        
        # prepare for dn loss
        if 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            output_known, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                assert len(targets[i]['boxes']) > 0
                # t = torch.range(0, len(targets[i]['labels']) - 1).long().to(self.device)
                t = torch.arange(0, len(targets[i]['labels'])).long().to(self.device)
                t = t.unsqueeze(0).repeat(scalar, 1)
                tgt_idx = t.flatten()
                output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(self.device).unsqueeze(1) + t
                output_idx = output_idx.flatten()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                if loss == 'scale_map':
                    continue
                l_dict.update(self.get_loss(loss, output_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        
        for loss in self.losses:           
            losses.update(self.get_loss(loss, outputs, targets, indices, num_valid_instances[loss]))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'scale_map' or loss == 'verts':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_valid_instances[loss])
                    l_dict = {f'{k}.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if 'dn_meta' in outputs:
                    if loss == 'scale_map':
                        continue
                    aux_outputs_known = output_known['aux_outputs'][i]
                    l_dict={}
                    for loss in self.losses:
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))
                    l_dict = {k + f'_dn.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # if 'scale_map' in outputs:
        #     enc_outputs = outputs['enc_outputs']
        #     indices = self.matcher.forward_enc(enc_outputs, targets)
        #     for loss in ['confs_enc', 'boxes_enc']:
        #         l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_valid_instances[loss.replace('_enc','')])
        #         l_dict = {k + f'_enc': v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        return losses

class SetCriterion_SATPR(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses = ['confs','boxes', 'poses','betas', 'j3ds','j2ds', 'depths', 'kid_offsets'], 
                focal_alpha=0.25, focal_gamma = 2.0, j2ds_norm_scale = 518, num_queries = 50):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        self.weight_dict = weight_dict
        

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None
        self.num_queries = num_queries


    def loss_boxes(self, loss, outputs, targets, indices, num_instances, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        assert loss == 'boxes'
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape
        
        src_boxes = src
        target_boxes = target

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['boxes'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['giou'] = loss_giou.sum() / num_instances

        # # calculate the x,y and h,w loss
        # with torch.no_grad():
        #     losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #     losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_next_boxes(self, loss, outputs, targets, indices, num_instances, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # assert 'pred_boxes' in outputs
        # assert loss == 'boxes'
        # idx = self._get_src_permutation_idx(indices)
        # valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        # if len(valid_idx) == 0:
        #     return {loss: torch.tensor(0.).to(self.device)}

        if outputs['next_bbox'] == None:
            return {'next_bbox': torch.tensor(0.).to(self.device), 'next_giou': torch.tensor(0.).to(self.device)}
        
        src_boxes = outputs['next_bbox'][0]
        target_boxes = targets[0]['boxes']

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        num_instances = targets[0]['pnum']
        losses = {}
        losses['next_bbox'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['next_giou'] = loss_giou.sum() / num_instances

        # # calculate the x,y and h,w loss
        # with torch.no_grad():
        #     losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #     losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses
        

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, indices, num_instances, **kwargs):
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape

        losses = {}
        loss_mask = None

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:45,:]
            target = target[:,:45,:]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([t['j2ds_mask'][i] for t, (_, i) in zip(targets, indices) if 'j2ds' in t], dim=0)
            # Use 45 smpl joints
            src = src[:,:45,:]
            target = target[:,:45,:]
            loss_mask = loss_mask[:,:45,:]
        
        valid_loss = torch.abs(src-target)

        # if loss == 'j2ds':
        #     print(src.shape)
        #     print(target.shape)
        #     print(num_instances)
        #     exit(0)
        
        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances



        return losses

    def loss_scale_map(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'scale_map'

        pred_map = outputs['enc_outputs']['scale_map']
        tgt_map = torch.cat([t['scale_map'] for t in targets], dim=0)
        assert pred_map.shape == tgt_map.shape

        labels = tgt_map[:,0]
        pred_scales = pred_map[:,1]
        tgt_scales = tgt_map[:, 1]

        detection_valid_mask = labels.bool()
        cur = 0
        lens = [len(t['scale_map']) for t in targets]
        for i, tgt in enumerate(targets):
            if tgt['detect_all_people']:
                detection_valid_mask[cur:cur+lens[i]] = True
            cur += lens[i]

     
        losses = {}
        losses['map_confs'] = focal_loss(pred_map[:,0], labels, valid_mask=detection_valid_mask)/1.
        losses['map_scales'] = torch.abs((pred_scales - tgt_scales)[torch.where(labels)[0]]).sum()/num_instances


        return losses

    def loss_confs(self, loss, outputs, targets, indices, num_instances, is_dn=False, **kwargs):
        assert loss == 'confs'
        idx = self._get_src_permutation_idx(indices)
        pred_confs = outputs['pred_'+loss]

        with torch.no_grad():
            labels = torch.zeros_like(pred_confs)
            labels[idx] = 1
            detection_valid_mask = torch.zeros_like(pred_confs,dtype=bool)
            detection_valid_mask[idx] = True
            valid_batch_idx = torch.where(torch.tensor([t['detect_all_people'] for t in targets]))[0]
            detection_valid_mask[valid_batch_idx] = True

        
        losses = {}
        if is_dn:
            losses[loss] = focal_loss(pred_confs, labels) / num_instances
        else:
            losses[loss] = focal_loss(pred_confs, labels, valid_mask = detection_valid_mask) / num_instances

        return losses

    def loss_normalized_depths(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'depths' 
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx][...,[1]]  # [d d/f]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)[...,[0]]
        target_focals = torch.cat([t['focals'][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)

        # print(src.shape, target.shape, target_focals.shape)

        src = target_focals * src
        
        assert src.shape == target.shape

        valid_loss = torch.abs(1./(src + 1e-8) - 1./(target + 1e-8))
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            'confs': self.loss_confs,
            'boxes': self.loss_boxes,
            'poses': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            'depths': self.loss_normalized_depths,
            'scale_map': self.loss_scale_map,       
            # 'next_pred_boxes': self.loss_next_boxes,
        }
        if loss != "next_pred_boxes":
            # assert loss in loss_map, f'do you really want to compute {loss} loss?'
            return loss_map[loss](loss, outputs, targets, indices, num_instances, **kwargs)


    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            if loss != 'scale_map':
                for t in targets:
                    num_instances += t['pnum'] if loss in t else 0
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            else:
                for t in targets:
                    num_instances += t['scale_map'][...,0].sum().item()
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        return num_valid_instances

    def prep_for_dn(self, dn_meta):
        output_known = dn_meta['output_known']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size//num_dn_groups

        return output_known, single_pad, num_dn_groups

    def _slice_outputs_for_range(self, outputs, q_start, q_end):
        out = {}
        for k, v in outputs.items():
            if k in ('aux_outputs', 'enc_outputs', 'sat'):
                continue
            if v is None:
                out[k] = None
                continue
            if k == 'pred_intrinsics' or k == 'dn_meta' or k == 'next_bbox' or k == 'kept_indices':
                out[k] = v
                continue
            out[k] = v[:, q_start:q_end, ...]
        return out


    def forward(self, _outputs, _targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        _losses = []
        for outputs, targets in zip(_outputs, _targets[0]):
            # remove invalid information in targets
            for t in targets:
                if not t['3d_valid']:
                    for key in ['betas', 'poses', 'j3ds', 'depths', 'focals']:
                        if key in t:
                            del t[key]

            # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
            # indices = self.matcher(outputs_without_aux, targets)

            total_query = outputs['pred_poses'].shape[1]
            outputs_pr = self._slice_outputs_for_range(outputs, 0, total_query - self.num_queries) 
            outputs_ref = self._slice_outputs_for_range(outputs, total_query - self.num_queries, total_query)
            outputs_ref['enc_outputs'] = outputs['enc_outputs']
            # # Retrieve the matching between the outputs of the last layer and the targets
            # pr_indices = self.matcher(outputs_pr, targets)
            ref_indices = self.matcher(outputs_ref, targets)
            pr_cnt = [target['pnum'] for target in targets]
            pr_indices = [(torch.arange(s), torch.arange(s)) for s in pr_cnt]
            self.device = outputs['pred_poses'].device
            num_valid_instances = self.get_valid_instances(targets)

            # Compute all the requested losses
            losses = {}
            
            # prepare for dn loss
            if 'dn_meta' in outputs:
                dn_meta = outputs['dn_meta']
                output_known, single_pad, scalar = self.prep_for_dn(dn_meta)

                dn_pos_idx = []
                dn_neg_idx = []
                for i in range(len(targets)):
                    assert len(targets[i]['boxes']) > 0
                    # t = torch.range(0, len(targets[i]['labels']) - 1).long().to(self.device)
                    t = torch.arange(0, len(targets[i]['labels'])).long().to(self.device)
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(self.device).unsqueeze(1) + t
                    output_idx = output_idx.flatten()

                    dn_pos_idx.append((output_idx, tgt_idx))
                    dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

                l_dict = {}
                for loss in self.losses:
                    if loss == 'scale_map' or loss == 'next_pred_boxes':
                        continue
                    l_dict.update(self.get_loss(loss, output_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))
                l_dict = {k + f'_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)

            # print(indices)
            
            for loss in self.losses:           
                # losses.update(self.get_loss(loss, outputs, targets, indices, num_valid_instances[loss]))
                if loss == 'next_pred_boxes':
                    continue
                losses.update(self.get_loss(loss, outputs_ref, targets, ref_indices, num_valid_instances[loss]))
                if loss == 'scale_map':
                    continue
                pr_loss = self.get_loss(loss, outputs_pr, targets, pr_indices, num_valid_instances[loss])
                pr_loss = {f'{k}_pr': v for k, v in pr_loss.items()}
                losses.update(pr_loss)
            # losses.update(self.loss_next_boxes('confs', outputs_ref, targets, ref_indices, num_valid_instances['confs']))
            
            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == 'scale_map' or loss == 'next_pred_boxes':
                            continue
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_valid_instances[loss])
                        l_dict = {f'{k}.{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                    if 'dn_meta' in outputs:
                        if loss == 'scale_map' or loss == 'next_pred_boxes':
                            continue
                        aux_outputs_known = output_known['aux_outputs'][i]
                        l_dict={}
                        for loss in self.losses:
                            l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))
                        l_dict = {k + f'_dn.{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
            
            _losses.append(losses)

            # if 'scale_map' in outputs:
            #     enc_outputs = outputs['enc_outputs']
            #     indices = self.matcher.forward_enc(enc_outputs, targets)
            #     for loss in ['confs_enc', 'boxes_enc']:
            #         l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_valid_instances[loss.replace('_enc','')])
            #         l_dict = {k + f'_enc': v for k, v in l_dict.items()}
            #         losses.update(l_dict)

        loss_sums = defaultdict(float)
        num_steps = len(_losses)

        for loss_dict in _losses:
            for k, v in loss_dict.items():
                loss_sums[k] += v

        _losses = {k: v / num_steps for k, v in loss_sums.items()}
        return _losses

class SetCriterion_SATPR_IMG(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses = ['confs','boxes', 'poses','betas', 'j3ds','j2ds', 'depths', 'kid_offsets'], 
                focal_alpha=0.25, focal_gamma = 2.0, j2ds_norm_scale = 518, num_queries = 50):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        self.weight_dict = weight_dict
        

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None
        self.num_queries = num_queries


    def loss_boxes(self, loss, outputs, targets, indices, num_instances, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        assert loss == 'boxes'
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape
        
        src_boxes = src
        target_boxes = target

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['boxes'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['giou'] = loss_giou.sum() / num_instances

        # # calculate the x,y and h,w loss
        # with torch.no_grad():
        #     losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #     losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, indices, num_instances, **kwargs):
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape

        losses = {}
        loss_mask = None

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:45,:]
            target = target[:,:45,:]
        elif loss == 'j2ds':
            # if targets[0]['ds'] == 'posetrack':
            #     target = torch.cat([t['ori_'+loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
            #     loss_mask = torch.cat([t['ori_'+loss+'_conf'][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
            #     src = src / self.j2ds_norm_scale
            #     # target = target / self.j2ds_norm_scale
            #     # Use 45 smpl joints
            #     src = src[:,:24,:]
            #     target = target[:,:24,:]
            #     loss_mask = loss_mask[:,:24].unsqueeze(-1).repeat(1,1,2)
            # else:
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([t['j2ds_mask'][i] for t, (_, i) in zip(targets, indices) if 'j2ds' in t], dim=0)
            # Use 45 smpl joints
            src = src[:,:45,:]
            target = target[:,:45,:]
            loss_mask = loss_mask[:,:45,:]

        # if loss == 'j2ds':
        #     if 'img' in outputs:
        #         from utils.transforms import unNormalize
        #         from utils.visualization import tensor_to_BGR, pad_img
        #         import numpy as np
        #         import cv2
        #         src = src * loss_mask
        #         target = target * loss_mask
        #         img = np.ascontiguousarray(tensor_to_BGR(unNormalize(outputs['img'][0]).cpu()))
        #         for idx in range(len(src)):
        #             for j, (x, y) in enumerate(src[idx].detach().cpu().numpy() * self.j2ds_norm_scale):
        #                 cv2.circle(img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)
        #                 cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                             1, (0,255,255), 1, cv2.LINE_AA)

        #             for j, (x, y) in enumerate(target[idx].cpu().numpy()  * self.j2ds_norm_scale):
        #                 cv2.circle(img, (int(round(x)), int(round(y))), 3, (255,0,0), -1, cv2.LINE_AA)
        #                 cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                             1, (255,0,0), 1, cv2.LINE_AA)
        #         cv2.imwrite("kp2d2.png", img)
        #         img = np.ascontiguousarray(tensor_to_BGR(unNormalize(outputs['img'][0]).cpu()))
        #         cv2.imwrite("kp2d2_img.png", img)
        #         import pdb; pdb.set_trace()
        # if loss == 'verts':
        #     import numpy as np
        #     from utils.visualization import vis_meshes_img, get_colors_rgb
        #     device = outputs['img'][0]
        #     mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        #     std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        #     img = outputs['img'][0] * std.to(device) + mean.to(device)
        #     img = torch.clamp(img, 0, 255).to(torch.uint8)
        #     img = img.cpu().numpy()
            
        #     img = np.ascontiguousarray(img.transpose(1,2,0))
        #     from models.human_models import SMPL_Layer
        #     from configs.paths import smpl_model_path
        #     import cv2
        #     human_model = SMPL_Layer(model_path = smpl_model_path, with_genders = False)
        #     idx = 0
        #     img_size = img.shape
        #     pred_idx = outputs['pred_confs'][idx] > 0.3
        #     pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()[pred_idx.detach().cpu().numpy()[:, 0]]
        #     colors = get_colors_rgb(len(pred_verts))
        #     pred_mesh_img = vis_meshes_img(img = img.copy(),
        #                                     verts = pred_verts,
        #                                     smpl_faces = human_model.faces,
        #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
        #                                     colors=colors)[:img_size[0],:img_size[1]]
        #     img_name = targets[0]['img_path'].split('/')[-1].split('.')[0]
        #     cv2.imwrite(os.path.join('test_train', f'trn_sam2_{img_name}.png'), pred_mesh_img)
        #     print(os.path.join('test_train', f'trn_sam2_{img_name}.png'))
        #     import pdb; pdb.set_trace()
        
        valid_loss = torch.abs(src-target)

        # if loss == 'j2ds':
        #     print(src.shape)
        #     print(target.shape)
        #     print(num_instances)
        #     exit(0)
        
        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances



        return losses

    def loss_scale_map(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'scale_map'

        pred_map = outputs['enc_outputs']['scale_map']
        tgt_map = torch.cat([t['scale_map'] for t in targets], dim=0)
        assert pred_map.shape == tgt_map.shape

        labels = tgt_map[:,0]
        pred_scales = pred_map[:,1]
        tgt_scales = tgt_map[:, 1]

        detection_valid_mask = labels.bool()
        cur = 0
        lens = [len(t['scale_map']) for t in targets]
        for i, tgt in enumerate(targets):
            if tgt['detect_all_people']:
                detection_valid_mask[cur:cur+lens[i]] = True
            cur += lens[i]

        pred_scale_map = pred_map[:,0]
        losses = {}
        losses['map_confs'] = focal_loss(pred_scale_map, labels, valid_mask=detection_valid_mask)/1.
        losses['map_scales'] = torch.abs((pred_scales - tgt_scales)[torch.where(labels)[0]]).sum()/num_instances

        return losses

    def loss_confs(self, loss, outputs, targets, indices, num_instances, is_dn=False, **kwargs):
        assert loss == 'confs'
        idx = self._get_src_permutation_idx(indices)
        pred_confs = outputs['pred_'+loss]

        with torch.no_grad():
            labels = torch.zeros_like(pred_confs)
            labels[idx] = 1
            detection_valid_mask = torch.zeros_like(pred_confs,dtype=bool)
            detection_valid_mask[idx] = True
            valid_batch_idx = torch.where(torch.tensor([t['detect_all_people'] for t in targets]))[0]
            detection_valid_mask[valid_batch_idx] = True

            # if 'is_pn' in kwargs and kwargs['is_pn']:
            #     # labels = torch.zeros_like(pred_confs)
            #     detection_valid_mask = torch.zeros_like(pred_confs,dtype=bool)
            #     for n, kept_idx in enumerate(outputs['kept_indices']):
            #         detection_valid_mask[n][:len(kept_idx)] = True
                
        losses = {}
        if is_dn:
            losses[loss] = focal_loss(pred_confs, labels) / num_instances
        else:
            losses[loss] = focal_loss(pred_confs, labels, valid_mask = detection_valid_mask) / num_instances
        return losses

    def loss_normalized_depths(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'depths' 
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device = self.device)*(loss in t) for t, (_, i) in zip(targets, indices)]))[0]
        
        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_'+loss][idx][valid_idx][...,[1]]  # [d d/f]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)[...,[0]]
        target_focals = torch.cat([t['focals'][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)

        # print(src.shape, target.shape, target_focals.shape)

        src = target_focals * src
        
        assert src.shape == target.shape

        valid_loss = torch.abs(1./(src + 1e-8) - 1./(target + 1e-8))
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            'confs': self.loss_confs,
            'boxes': self.loss_boxes,
            'poses': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            'verts': self.loss_L1,
            'depths': self.loss_normalized_depths,
            'scale_map': self.loss_scale_map,       
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if len(indices) == 0:
            if loss == "scale_map":
                return {'map_scales': torch.zeros(1, dtype=torch.float32, device=self.device), 
                        'map_confs': torch.zeros(1, dtype=torch.float32, device=self.device)}
            else:
                return {loss: torch.zeros(1, dtype=torch.float32, device=self.device)}
        else:
            return loss_map[loss](loss, outputs, targets, indices, num_instances, **kwargs)

    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            if loss != 'scale_map':
                for t in targets:
                    num_instances += t['pnum'] if loss in t else 0
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            else:
                for t in targets:
                    num_instances += t['scale_map'][...,0].sum().item()
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        return num_valid_instances

    def prep_for_dn(self, dn_meta):
        output_known = dn_meta['output_known']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size//num_dn_groups

        return output_known, single_pad, num_dn_groups

    def _slice_outputs_for_range(self, outputs, q_start, q_end):
        out = {}
        for k, v in outputs.items():
            if k in ('aux_outputs', 'enc_outputs', 'sat'):
                continue
            if v is None:
                out[k] = None
                continue
            if k == 'pred_intrinsics' or k == 'dn_meta' or k == 'pr_dn_meta' or k == 'img' or k == 'kept_indices':
                out[k] = v
                continue
            out[k] = v[:, q_start:q_end, ...]
        return out


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # remove invalid information in targets
        for t in targets:
            if not t['3d_valid']:
                for key in ['betas', 'poses', 'j3ds', 'depths', 'focals', 'img']:
                    if key in t:
                        del t[key]

        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
        # indices = self.matcher(outputs_without_aux, targets)

        total_query = outputs['pred_poses'].shape[1]
        outputs_pr = self._slice_outputs_for_range(outputs, 0, total_query - self.num_queries) 
        outputs_ref = self._slice_outputs_for_range(outputs, total_query - self.num_queries, total_query)
        outputs_ref['enc_outputs'] = outputs['enc_outputs']
        # # Retrieve the matching between the outputs of the last layer and the targets
        # pr_indices = self.matcher(outputs_pr, targets)
        # pr_cnt = [target['pnum'] for target in targets]
        # pr_indices = [(torch.arange(s), torch.arange(s)) for s in pr_cnt]
        kept_indices = outputs['kept_indices']
        pr_indices = [(torch.arange(len(idx)), idx) for idx in kept_indices]
        ref_indices = self.matcher(outputs_ref, targets)
        self.device = outputs['pred_poses'].device
        num_valid_instances = self.get_valid_instances(targets)

        # Compute all the requested losses
        losses = {}
        
        # prepare for dn loss
        if 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            output_known, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                assert len(targets[i]['boxes']) > 0
                # t = torch.range(0, len(targets[i]['labels']) - 1).long().to(self.device)
                t = torch.arange(0, len(targets[i]['labels'])).long().to(self.device)
                t = t.unsqueeze(0).repeat(scalar, 1)
                tgt_idx = t.flatten()
                output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(self.device).unsqueeze(1) + t
                output_idx = output_idx.flatten()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                if loss == 'scale_map' or loss == 'verts':
                    continue
                l_dict.update(self.get_loss(loss, output_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        
        # if 'pr_dn_meta' in outputs:
        #     dn_meta = outputs['pr_dn_meta']
        #     output_known, single_pad, scalar = self.prep_for_dn(dn_meta)

        #     dn_pos_idx = []
        #     dn_neg_idx = []
        #     for i in range(len(targets)):
        #         assert len(targets[i]['boxes']) > 0
        #         # t = torch.range(0, len(targets[i]['labels']) - 1).long().to(self.device)
        #         t = torch.arange(0, len(targets[i]['labels'])).long().to(self.device)
        #         t = t.unsqueeze(0).repeat(scalar, 1)
        #         tgt_idx = t.flatten()
        #         output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(self.device).unsqueeze(1) + t
        #         output_idx = output_idx.flatten()

        #         dn_pos_idx.append((output_idx, tgt_idx))
        #         dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

        #     l_dict = {}
        #     for loss in self.losses:
        #         if loss == 'scale_map':
        #             continue
        #         l_dict.update(self.get_loss(loss, output_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))

        #     l_dict = {k + f'_pr': v for k, v in l_dict.items()}
        #     losses.update(l_dict)

        # print(indices)
        
        for loss in self.losses:           
            # losses.update(self.get_loss(loss, outputs, targets, indices, num_valid_instances[loss]))
            losses.update(self.get_loss(loss, outputs_ref, targets, ref_indices, num_valid_instances[loss]))
            if loss == 'scale_map':
                continue
            pr_loss = self.get_loss(loss, outputs_pr, targets, pr_indices, num_valid_instances[loss], is_pn=True)
            pr_loss = {f'{k}_pr': v for k, v in pr_loss.items()}
            losses.update(pr_loss)
            
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'scale_map' or loss == 'verts':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_valid_instances[loss])
                    l_dict = {f'{k}.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if 'dn_meta' in outputs:
                    if loss == 'scale_map' or loss == 'verts':
                        continue
                    aux_outputs_known = output_known['aux_outputs'][i]
                    l_dict={}
                    for loss in self.losses:
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_valid_instances[loss]*scalar, is_dn=True))
                    l_dict = {k + f'_dn.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # if 'scale_map' in outputs:
        #     enc_outputs = outputs['enc_outputs']
        #     indices = self.matcher.forward_enc(enc_outputs, targets)
        #     for loss in ['confs_enc', 'boxes_enc']:
        #         l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_valid_instances[loss.replace('_enc','')])
        #         l_dict = {k + f'_enc': v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        return losses

class SetCriterionSAM2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  weight_dict, losses = ['poses','betas', 'j3ds','j2ds', 'depths'], 
                focal_alpha=0.25, focal_gamma = 2.0, j2ds_norm_scale = 518):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        self.weight_dict = weight_dict
        

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, num_instances, **kwargs):
        total_loss = 0.0
        denom = 0.0  

        device = self.device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device) / num_instances

        losses = {}
        loss_mask = None

        # output_key = 'pred_'+loss
        # if loss == 'j3ds' or loss == 'j2ds':
        #     output_pnums = [out[0].shape[0] for out in outputs['pred_'+loss]]
        # else:
        # output_pnums = [out.shape[0] for out in outputs['pred_'+loss]]
        # target_pnums = [t['pnum'] for t in targets]

        # print(output_pnums, target_pnums)
        if loss == 'root_pose':
            src = [ outputs['pred_poses'][b][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'poses' in targets[b]]
        else:
            src = [ outputs['pred_'+loss][b][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]]

        if len(src) == 0:
            losses[loss] = total_loss.squeeze(0)
            return losses

        src = torch.cat(src, dim=0)
        if loss == 'root_pose':
            target = torch.cat([targets[b]['poses'][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'poses' in targets[b]], dim=0)
        else:
            target = torch.cat([targets[b][loss][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]], dim=0)

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]

            # from utils.constants import H36M_EVAL_JOINTS
            # from utils.evaluation import cal_3d_position_error
            # import pdb; pdb.set_trace()
            # gt_pelvis = src[0][[0],:].detach().cpu().numpy().copy()
            # pred_pelvis = target[0][[0],:].detach().cpu().numpy().copy()

            # gt3d = (src[0].detach().cpu().numpy() - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
            # pred3d = (target[0].detach().cpu().numpy() - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
            # error_j, _ = cal_3d_position_error(pred3d, gt3d)
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([
                targets[b]['j2ds_mask'][:targets[b]['pnum'], :24, :]
                for b in range(len(targets)) if targets[b]['pnum'] > 0
            ], dim=0)
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
            loss_mask = loss_mask[:,:24,:]
        # elif loss == 'poses':
        #     src = src[:,3:]
        #     target = target[:,3:]
        # elif loss == 'root_pose':
        #     src = src[:,:3]
        #     target = target[:,:3]

        # if loss == 'verts':
        #     import numpy as np
        #     from utils.visualization import vis_meshes_img, get_colors_rgb
        #     device = outputs['img'][0]
        #     mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        #     std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        #     img = outputs['img'][0] * std.to(device) + mean.to(device)
        #     img = torch.clamp(img, 0, 255).to(torch.uint8)
        #     img = img.cpu().numpy()
            
        #     img = np.ascontiguousarray(img.transpose(1,2,0))
        #     from models.human_models import SMPL_Layer
        #     from configs.paths import smpl_model_path
        #     import cv2
        #     human_model = SMPL_Layer(model_path = smpl_model_path, with_genders = False)
        #     idx = 0
        #     img_size = img.shape
        #     pred_verts = outputs['pred_verts'][idx].detach().cpu().numpy()
        #     colors = get_colors_rgb(len(pred_verts))
        #     pred_mesh_img = vis_meshes_img(img = img.copy(),
        #                                     verts = pred_verts,
        #                                     smpl_faces = human_model.faces,
        #                                     cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
        #                                     colors=colors)[:img_size[0],:img_size[1]]
        #     img_name = targets[0]['img_path'].split('/')[-1].split('.')[0]
        #     cv2.imwrite(os.path.join('test_train', f'trn_sam2_{img_name}.png'), pred_mesh_img)
        #     import pdb; pdb.set_trace()
        # if loss == 'j2ds':
        #     from utils.transforms import unNormalize
        #     from utils.visualization import tensor_to_BGR, pad_img
        #     import numpy as np
        #     import cv2
        #     device = outputs['img'][0]
        #     mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        #     std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        #     img = outputs['img'][0] * std.to(device) + mean.to(device)
        #     img = torch.clamp(img, 0, 255).to(torch.uint8)
        #     img = img.cpu().numpy()
            
        #     img = np.ascontiguousarray(img.transpose(1,2,0))
        #     for j, (x, y) in enumerate(src[0].detach().cpu().numpy() * self.j2ds_norm_scale):
        #         cv2.circle(img, (int(round(x)), int(round(y))), 3, (0,255,0), -1, cv2.LINE_AA)
        #         cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (0,255,255), 1, cv2.LINE_AA)

        #     for j, (x, y) in enumerate(target[0].cpu().numpy()  * self.j2ds_norm_scale):
        #         cv2.circle(img, (int(round(x)), int(round(y))), 3, (255,0,0), -1, cv2.LINE_AA)
        #         cv2.putText(img, str(j), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (255,0,0), 1, cv2.LINE_AA)
        #     cv2.imwrite("kp2d_test.jpg", img)
        #     import pdb; pdb.set_trace()
        # # elif loss == 'j3ds':
        # #     import pdb; pdb.set_trace()

        valid_loss = torch.abs(src-target)

        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        total_loss = valid_loss.flatten(1).mean(-1).sum()/num_instances
        losses[loss] = total_loss
        return losses

    def loss_L2(self, loss, outputs, targets, num_instances, **kwargs):
        total_loss = 0.0
        denom = 0.0  # num_instances가 없으면 여기서 계산

        device = self.device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device) / num_instances

        losses = {}
        loss_mask = None

        # output_key = loss
        # if loss == 'j3ds' or loss == 'j2ds':
        #     output_pnums = [out[0].shape[0] for out in outputs['pred_'+loss]]
        # else:

        src = [ outputs[ loss][b][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]]

        if len(src) == 0:
            losses[loss] = total_loss.squeeze(0)
            return losses
        
        src = torch.cat(src, dim=0)
        target = torch.cat([targets[b][loss][:targets[b]['pnum']] for b in range(len(targets)) if targets[b]['pnum'] > 0 and loss in targets[b]], dim=0)

        if loss == 'j3ds':
            # Root aligned
            src = src - src[...,[0],:].clone()
            target = target - target[...,[0],:].clone()
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([
                targets[b]['j2ds_mask'][:targets[b]['pnum'], :24 :]
                for b in range(len(targets)) if targets[b]['pnum'] > 0
            ], dim=0)
            # Use 45 smpl joints
            src = src[:,:24,:]
            target = target[:,:24,:]
            loss_mask = loss_mask[:,:24,:]
        elif loss == 'poses':
            src = src.view(src.shape[0], -1)
            target = target[:,:src.shape[1]]

            # from utils.transforms import rotation_matrix_to_angle_axis
            # from models.geometry import matrix_to_axis_angle
        
        # valid_loss = torch.abs(src-target)
        valid_loss = F.mse_loss(src, target, reduction='none')

        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        # if loss == 'betas':
        #     valid_loss = valid_loss*self.betas_weight.to(src.device)
        
        total_loss = valid_loss.flatten(1).mean(-1).sum()/num_instances
        losses[loss] = total_loss
        return losses

    def loss_normalized_depths(self, loss, outputs, targets, num_instances, **kwargs):
        assert loss == 'depths' 
        losses = {}
        device = self.device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device) 
        
        src = [outputs['pred_'+loss][b][:targets[b]['pnum'], [1]] 
               for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'depths' in targets[b]]

        if len(src) == 0:
            losses[loss] = total_loss.squeeze(0)
            return losses
        src = torch.cat(src, dim=0)

        target  = torch.cat([targets[b]['depths'][:targets[b]['pnum'], [0]] 
               for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'depths' in targets[b]], dim=0)
        target_focals    = torch.cat([targets[b]['focals'][:targets[b]['pnum'], [0]] 
               for b in range(len(targets)) if targets[b]['pnum'] > 0 and 'focals' in targets[b]], dim=0)
        
        assert src.shape == target.shape
        src = target_focals * src

        valid_loss = torch.abs(1./(src + 1e-8) - 1./(target + 1e-8))
        losses[loss] = valid_loss.flatten(1).mean(-1).sum()/num_instances
        return losses

    def get_loss(self, loss, outputs, targets, num_instances, **kwargs):
        loss_map = {
            # 'confs': self.loss_confs,
            'poses': self.loss_L1,
            # 'root_pose': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            'verts': self.loss_L1,
            # 'poses': self.loss_L2,
            # 'betas': self.loss_L2,
            # 'j3ds': self.loss_L2,
            # 'j2ds': self.loss_L2,
            # 'verts': self.loss_L2,
            'depths': self.loss_normalized_depths,
            # 'scale_map': self.loss_scale_map,       
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](loss, outputs, targets, num_instances, **kwargs)


    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            if loss != 'scale_map':
                for t in targets:
                    num_instances += t['pnum'] if loss in t else 0
                num_instances = torch.as_tensor([num_instances], dtype=torch.float32, device=self.device)
            else:
                for t in targets:
                    num_instances += t['scale_map'][...,0].sum().item()
                num_instances = torch.as_tensor([num_instances], dtype=torch.float32, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        return num_valid_instances
        
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # remove invalid information in targets
        for t in targets:
            if not t['3d_valid']:
                for key in ['betas', 'poses', 'j3ds', 'depths', 'focals']:
                    if key in t:
                        del t[key]

        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        self.device = outputs['pred_poses'][0].device
        num_valid_instances = self.get_valid_instances(targets)

        # Compute all the requested losses
        losses = {k: torch.zeros((), device=self.device, dtype=torch.float32) for k in self.losses}

        # outputs = {
        #     k: torch.stack([d[k] for d in outputs], dim=0)
        #     if isinstance(outputs[0][k], torch.Tensor)
        #     else [d[k] for d in outputs]
        #     for k in outputs[0]
        # }
        
        # outputs['verts'] = outputs['smpl_vertices']
        for loss_name in self.losses:
            if loss_name == "root_pose":
                out = self.get_loss(loss_name, outputs, targets, num_valid_instances['poses'])
            else:
                out = self.get_loss(loss_name, outputs, targets, num_valid_instances[loss_name])
            if loss_name in out:
                losses[loss_name] = losses[loss_name] + out[loss_name]

        
        return losses