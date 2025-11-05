# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, 
                cost_conf: float = 1, 
                cost_bbox: float = 1, 
                cost_giou: float = 1,
                cost_kpts: float = 10, 
                j2ds_norm_scale: float = 518,
                ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_conf = cost_conf
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_kpts = cost_kpts
        self.j2ds_norm_scale = j2ds_norm_scale
        assert cost_conf != 0 or cost_bbox != 0 or cost_giou != 0 or cost_kpts != 0, "all costs cant be 0"

        # self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert outputs['pred_confs'].shape[0]==len(targets)
        bs, num_queries, _ = outputs["pred_confs"].shape

        # We flatten to compute the cost matrices in a batch
        out_conf = outputs['pred_confs'].flatten(0,1)  # [batch_size * num_queries, 1]
        out_bbox = outputs["pred_boxes"].flatten(0,1)  # [batch_size * num_queries, 4]
        out_kpts = outputs['pred_j2ds'][...,:22,:].flatten(2).flatten(0,1) / self.j2ds_norm_scale

        # Also concat the target labels and boxes
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_kpts = torch.cat([v['j2ds'][:,:22,:].flatten(1) for v in targets]) / self.j2ds_norm_scale
        tgt_kpts_mask = torch.cat([v['j2ds_mask'][:,:22,:].flatten(1) for v in targets])
        tgt_kpts_vis_cnt = tgt_kpts_mask.sum(-1)
        assert (torch.all(tgt_kpts_vis_cnt))

        # Compute the confidence cost.
        alpha = 0.25
        gamma = 2.0
        cost_conf = alpha * ((1 - out_conf) ** gamma) * (-(out_conf + 1e-8).log())
        # cost_conf = -(out_conf+1e-8).log()

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the mean L1 cost between visible joints
        all_dist = torch.abs(out_kpts[:,None,:] - tgt_kpts[None,:,:])
        mean_dist = (all_dist * tgt_kpts_mask[None,:,:]).sum(-1) / tgt_kpts_vis_cnt[None,:]
        cost_kpts = mean_dist

        try:
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_conf*cost_conf + self.cost_kpts*cost_kpts + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        except:
            return []
        # return []


def build_matcher(args):
    return HungarianMatcher(
        cost_conf=args.set_cost_conf, 
        cost_bbox=args.set_cost_bbox, 
        cost_giou=args.set_cost_giou, 
        cost_kpts=args.set_cost_kpts,
        j2ds_norm_scale=args.input_size
    )

@torch.no_grad()
def align_object_order_across_frames(
    meta_seq,
    j2ds_norm_scale: float = 518.0,
    w_kpts: float = 10.0,
    w_bbox: float = 1.0,
    w_giou: float = 1.0,
    inplace: bool = True,
):
    def _to_tensor(x):
        return torch.as_tensor(x).float()

    def _pack(objs, k_use=22):
        N = len(objs)
        if N == 0:
            return (torch.empty(0,4), torch.empty(0,22,2), torch.empty(0,22,2))
        boxes = torch.stack([_to_tensor(o['boxes']) for o in objs], dim=0)  # [N,4] (cx,cy,w,h)
        if 'j2ds' in objs[0]:
            K = min(k_use, _to_tensor(objs[0]['j2ds']).shape[0])
            j2ds  = torch.stack([_to_tensor(o['j2ds'])[:K]  for o in objs], dim=0)  # [N,K,2]
            if 'j2ds_mask' in objs[0]:
                jmask = torch.stack([_to_tensor(o['j2ds_mask'])[:K] for o in objs], dim=0)  # [N,K,2]
            else:
                jmask = torch.ones_like(j2ds)
        else:
            K = 0
            j2ds  = torch.empty(N,0,2)
            jmask = torch.empty(N,0,2)
        return boxes, j2ds, jmask

    def _cost_matrix(cur_objs, prev_objs):
        cur_b, cur_j, cur_m = _pack(cur_objs)
        prev_b, prev_j, prev_m = _pack(prev_objs)

        Nc, Np = cur_b.shape[1], prev_b.shape[1]
        if Nc == 0 or Np == 0:
            return torch.zeros(Nc, Np)

        # bbox L1
        cost_bbox = torch.cdist(cur_b, prev_b, p=1)  # [Nc,Np]

        try:
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(cur_b)[0],
                box_cxcywh_to_xyxy(prev_b)[0],
            )
            cost_giou = -giou[None]
        except Exception:
            cost_giou = torch.zeros_like(cost_bbox)

        if cur_j.numel() > 0 and prev_j.numel() > 0:
            # [Nc,K,2] -> [Nc,2K]
            cur_flat  = (cur_j.reshape(Nc, -1)  / j2ds_norm_scale)
            prev_flat = (prev_j.reshape(Np, -1) / j2ds_norm_scale)
            prev_mflat = prev_m.reshape(Np, -1)
            vis_cnt = prev_mflat.sum(-1).clamp_min(1.0)  # [Np]

            # [Nc,Np,2K]
            all_dist = (cur_flat[:, None, :] - prev_flat[None, :, :]).abs()
            cost_kpts = (all_dist * prev_mflat[None, :, :]).sum(-1) / vis_cnt[None, :]
        else:
            cost_kpts = torch.zeros_like(cost_bbox)

        C = w_bbox*cost_bbox + w_giou*cost_giou + w_kpts*cost_kpts[None]
        return C

    seq = meta_seq if inplace else [list(frame) for frame in meta_seq]
    if len(seq) <= 1:
        return seq

    prev = seq[0]
    for t in range(1, len(seq)):
        cur = list(seq[t])

        C = _cost_matrix(cur, prev).cpu().numpy()  # rows=cur, cols=prev
        if C.size == 0:
            seq[t] = cur
            prev = seq[t]
            continue

        r_idx, c_idx = linear_sum_assignment(C[0])  # cur i -> prev j 매칭
        new_cur = [None] * len(prev)
        used_rows = set()
        for r, c in zip(r_idx, c_idx):
            if c < len(new_cur):
                new_cur[c] = cur[r]
            used_rows.add(r)

        for i in range(len(cur)):
            if i not in used_rows:
                new_cur.append(cur[i])

        new_cur = [o for o in new_cur if o is not None]

        seq[t] = new_cur
        prev = seq[t]

    return seq
