import torch 
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, nms_xyxy, box_iou_xyxy
import torch.nn.functional as F

class TorchKalman:
    """
    Linear KF on torch tensors
    state x = [cx, cy, w, h, vx, vy, vw, vh]^T (8,)
    meas  z = [cx, cy, w, h]^T (4,)
    Constant-velocity model with dt.
    """
    def __init__(self, dt=1.0, device="cpu", q_pos=1e-2, q_vel=1e-1, r_meas=1.0):
        self.device = torch.device(device)
        self.dt = dt

        F = torch.eye(8, device=self.device)
        F[0,4] = dt; F[1,5] = dt; F[2,6] = dt; F[3,7] = dt
        H = torch.zeros((4,8), device=self.device)
        H[0,0]=H[1,1]=H[2,2]=H[3,3]=1.0

        Q = torch.diag(torch.tensor(
            [q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel],
            device=self.device))
        R = torch.diag(torch.full((4,), float(r_meas), device=self.device))

        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x = torch.zeros(8, device=self.device)     # state
        self.P = torch.eye(8, device=self.device)       # covariance

        # Init covariance scales (위치/크기 vs 속도)
        self.P[0:4,0:4] *= 10.0
        self.P[4:8,4:8] *= 1000.0

    @torch.no_grad()
    def init_from_xyxy(self, box_xyxy):
        # set x[:4] from bbox, zero velocities
        z = box_xyxy_to_cxcywh(box_xyxy.unsqueeze(0)).squeeze(0)
        self.x[:4] = z
        self.x[4:] = 0.0

    @torch.no_grad()
    def predict(self):
        # x = F x ; P = F P F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # clamp sizes
        self.x[2] = torch.clamp(self.x[2], min=1.0)
        self.x[3] = torch.clamp(self.x[3], min=1.0)
        return self.get_xyxy()

    @torch.no_grad()
    def update(self, meas_cxcywh):
        # z residual
        z = meas_cxcywh
        y = z - (self.H @ self.x)                             # (4,)
        S = self.H @ self.P @ self.H.T + self.R               # (4,4)
        K = self.P @ self.H.T @ torch.linalg.inv(S)           # (8,4)
        self.x = self.x + K @ y
        I = torch.eye(8, device=self.x.device)
        self.P = (I - K @ self.H) @ self.P
        # clamp sizes
        self.x[2] = torch.clamp(self.x[2], min=1.0)
        self.x[3] = torch.clamp(self.x[3], min=1.0)
        return self.get_xyxy()

    @torch.no_grad()
    def get_xyxy(self):
        b = self.x[:4].unsqueeze(0)                           # (1,4) cxcywh
        return box_cxcywh_to_xyxy(b).squeeze(0) 

_link_ref = torch.tensor(
    [
        (23, 24),  # eyes
        (22, 16),  # nose -> shoulder
        (22, 17),  # nose -> shoulder
        (16, 17),  # shoulders
        (16, 1),  # left shoulder -> hip
        (17, 2),  # right shoulder -> hip
        (16, 18),  # left shoulder -> elbow
        (17, 19),  # right shoulder -> elbow
        (20, 18),  # left wrist -> elbow
        (21, 19),  # right wrist -> elbow
        (4, 1),  # left knee -> hip
        (5, 2),  # right knee -> hip
        (4, 7),  # left knee -> ankle
        (5, 8),  # right knee -> ankle
    ]
).T

_link_dists_ref = [
    0.07,
    0.3,
    0.3,
    0.35,
    0.55,
    0.55,
    0.26,
    0.26,
    0.25,
    0.25,
    0.38,
    0.38,
    0.4,
    0.4,
]

def normalized_weighted_score(
    kp0,
    c0,
    kp1,
    c1,
    std=0.08,
    return_dists=False,
    ref_scale=None,
    min_scale=0.02,
    max_scale=1,
    fixed_scale=None,
):
    """Measure similarity of two sets of body pose keypoints.

    This is a modified version of the COCO object keypoint similarity (OKS)
    calculation using a Cauchy distribution instead of a Gaussian. We also
    compute a normalizing scale on the fly based on projected limb proportions.
    """
    # Combine confidence terms
    # conf: ... num_people x num_people x num_points
    conf = (c0.unsqueeze(-2) * c1.unsqueeze(-3)) ** 0.5

    # Calculate scale adjustment
    scale0 = _get_skeleton_scale(kp0, c0)
    scale1 = _get_skeleton_scale(kp1, c1)
    if ref_scale is None:
        scale = scale0.unsqueeze(-1).maximum(scale1.unsqueeze(-2))
    elif ref_scale == 0:
        scale = scale0.unsqueeze(-1)
    elif ref_scale == 1:
        scale = scale1.unsqueeze(-2)

    # Set scale bounds
    zero_mask = scale == 0
    scale.clamp_(min_scale, max_scale)
    scale[zero_mask] = 1e-6
    if fixed_scale is not None:
        scale[:] = fixed_scale

    # Scale-adjusted distance calculation
    # kp: ... num_people x num_pts x 2
    # dists: ... num_people x num_people x num_pts
    dists = (kp0.unsqueeze(-3) - kp1.unsqueeze(-4)).norm(dim=-1)
    dists = dists / scale.unsqueeze(-1)

    total_conf = conf.sum(-1)
    zero_filt = total_conf == 0
    scores = 1 / (1 + (dists / std) ** 2)
    scores = (scores * conf).sum(-1)
    scores = scores / total_conf
    scores[zero_filt] = 0

    if return_dists:
        return scores, dists
    else:
        return scores

def _get_skeleton_scale(kps: torch.Tensor, valid=None):
    """Return approximate "size" of person in image.

    Instead of bounding box dimensions, we compare pairs of keypoints as defined
    in the links above. We ignore any distances that involve an "invalid" keypoint.
    We assume any values set to (0, 0) are invalid and should be ignored.
    """
    invalid = (kps == 0).all(-1)
    if valid is not None:
        invalid |= ~(valid > 0)

    # Calculate link distances
    dists = (kps[..., _link_ref[0], :] - kps[..., _link_ref[1], :]).norm(dim=-1)

    # Compare to reference distances
    ratio = dists / torch.tensor(_link_dists_ref, device=dists.device)

    # Zero out any invalid links
    invalid = invalid[..., _link_ref[0]] | invalid[..., _link_ref[1]]
    ratio *= (~invalid).float()

    # Return max ratio which corresponds to limb with least foreshortening
    max_ratio = ratio.max(-1)[0]
    return max_ratio.clamp_min(0.001)


class QueryTracker:
    def __init__(self,
                 conf_thresh=0.5,
                 iou_match_thresh=0.5,
                 iou_new_thresh=0.5,
                 max_age=15,
                 boxes_format="xyxy",
                 pr_conf_thresh=0.9,
                 is_add_new=True,
                 is_size_filter=True,
                 nms_threshold=0.7,
                 dt=1.0):
        self.conf_thresh = conf_thresh
        self.iou_match_thresh = iou_match_thresh
        self.iou_new_thresh = iou_new_thresh
        self.max_age = max_age
        assert boxes_format in ("xyxy", "cxcywh")
        self.boxes_format = boxes_format
        self.pr_conf_thresh=pr_conf_thresh
        self.is_add_new = is_add_new
        self.is_size_filter = is_size_filter
        self.nms_threshold = nms_threshold
        self.dt = dt

        self.next_id = 0
        self.tracks = {}
        self.prev_query_num = 3

    def reset(self):
        self.next_id = 0
        self.tracks = {}

    def _to_xyxy(self, boxes):
        if boxes.numel() == 0:
            return boxes
        if self.boxes_format == "xyxy":
            return boxes
        return box_cxcywh_to_xyxy(boxes)

    @torch.no_grad()
    def init_frame(self, boxes, confs, query_indices, kps=None, querys=None, nms_iou=0.3, gt_boxes=None):
        if boxes.numel() == 0:
            return []

        boxes_xyxy = boxes
        if self.is_add_new:
            # posetrack
            # valid = confs > self.conf_thresh
            # valid = confs > self.conf_thresh 
            valid = confs > self.conf_thresh + 0.2
        else:
            valid = confs > self.conf_thresh 
        boxes_xyxy = boxes_xyxy[valid]
        confs = confs[valid]
        query_indices  = query_indices[valid]
        kps = kps[valid]
        querys = querys[valid]

        keep = nms_xyxy(boxes_xyxy, confs, iou_thresh=nms_iou)
        if keep.numel() == 0:
            return []

        keep_boxes = boxes_xyxy[keep]
        keep_confs = confs[keep]
        keep_qidx  = query_indices[keep]
        keep_kps = kps[keep]
        keep_query = querys[keep]

        if gt_boxes is not None:
            gt = gt_boxes.to(keep_boxes.device).float()  # [N, 4]
            kb = keep_boxes  # [M, 4]

            x1 = torch.maximum(kb[:, None, 0], gt[None, :, 0])
            y1 = torch.maximum(kb[:, None, 1], gt[None, :, 1])
            x2 = torch.minimum(kb[:, None, 2], gt[None, :, 2])
            y2 = torch.minimum(kb[:, None, 3], gt[None, :, 3])

            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_k = ((kb[:, 2] - kb[:, 0]) * (kb[:, 3] - kb[:, 1]))[:, None]
            area_g = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))[None, :]
            union = area_k + area_g - inter + 1e-6

            ious = inter / union  # [M, N]
            best_iou, best_gt_idx = ious.max(dim=0)
            # max_idx = best_iou.argmax()
            # max_idx = best_gt_idx[:len(gt_boxes)]
            keep_boxes = keep_boxes[best_gt_idx]
            keep_confs = keep_confs[best_gt_idx]
            keep_qidx = keep_qidx[best_gt_idx]
            keep_kps = keep_kps[best_gt_idx]
            keep_query = keep_query[best_gt_idx]

        active_ids = []
        device = keep_boxes.device
        for b, c, q, k, query in zip(keep_boxes, keep_confs, keep_qidx, keep_kps, keep_query):
            if c < self.conf_thresh:
                continue
            tid = self.next_id
            self.next_id += 1
            
            # Initialize Kalman Filter
            kf = TorchKalman(dt=self.dt, device=device)
            kf.init_from_xyxy(b)
            
            # Original code (without Kalman Filter):
            # self.tracks[tid] = {
            #     "box": b.detach(),
            #     "age": 0,
            #     "query_idx": int(q),   
            #     "origin_qidx": int(q),
            #     "kps": k.detach(),
            #     "query": query.detach(),
            # }
            
            self.tracks[tid] = {
                "box": b.detach(),
                "age": 0,
                "query_idx": int(q),   
                "origin_qidx": int(q),
                "kps": k.detach(),
                "query": query.detach(),
                "kf": kf,
            }
            active_ids.append(tid)
        return active_ids

    @torch.no_grad()
    def update(self, boxes, confs, query_indices, kp2ds=None, querys=None, gt_boxes=None, overlap_iou=0.8):
        """
        Update policy:
        1) Update tracks via greedy IoU matching (save prev_box/prev_query just before applying the match).
        2) If updated tracks overlap each other with IoU >= overlap_iou, compute each track's
        self-consistency IoU = IoU(prev_box, curr_box); keep the one with larger value (i.e., moved less),
        and rollback the others' updates for this frame (only age++ for them).
        3) Drop stale tracks and return results.
        """
        boxes_xyxy = self._to_xyxy(boxes)

        # --- 0) Snapshots for rollback (state before this frame's update) & age++ baseline ---
        prev_box = {}
        prev_age = {}
        prev_qid = {}
        prev_kps = {}
        for tid in list(self.tracks.keys()):
            prev_box[tid] = self.tracks[tid]["box"].clone()
            prev_age[tid] = self.tracks[tid]["age"]
            prev_qid[tid] = self.tracks[tid]["query_idx"]
            prev_qid[tid] = self.tracks[tid]["kps"]
            prev_qid[tid] = self.tracks[tid]["query"]
            # default +1; will reset to 0 if matched
            self.tracks[tid]["age"] += 1

        widths  = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        min_w, min_h = 50, 50  

        # valid = confs >= self.conf_thresh
        # valid_idx = torch.nonzero(valid, as_tuple=True)[0]
        # valid_idx = torch.where(valid)[0]
        
        active_matched_ids = [tid for tid in self.tracks.keys() if self.tracks[tid]["age"] == 1]
        if self.is_size_filter:
            size_valid = (widths >= min_w) & (heights >= min_h)
            valid = (confs >= self.conf_thresh) & size_valid
            valid[:len(active_matched_ids)] = False
            pr_valid = (confs >= self.pr_conf_thresh) & size_valid
            pr_valid[len(active_matched_ids):] = False
        else:
            valid = confs >= self.conf_thresh
            valid[:len(active_matched_ids)] = False
            pr_valid = confs >= self.pr_conf_thresh
            pr_valid[len(active_matched_ids):] = False

        # print(confs[:len(active_matched_ids)])

        valid = pr_valid | valid
        # print(active_matched_ids, valid)
        # valid = confs >= self.conf_thresh
        # boxes_v_xyxy   = boxes_xyxy[valid]
        # # boxes_v_cxcywh = boxes_cxcywh[valid]
        # confs_v        = confs[valid]
        # qidx_v         = query_indices[valid]

        # --- 1) Keep only valid detections by confidence ---
        boxes_v = boxes_xyxy[valid]
        # Original code:
        # confs_v = confs[valid]
        # qidx_v = query_indices[valid]
        # kp2ds_v = kp2ds[valid]
        # querys = querys[0]
        # querys_v = querys[valid]
        # print(self.frame_id,confs)
        
        # Added for Kalman Filter (cxcywh format needed for KF update)
        boxes_v_cxcywh = box_xyxy_to_cxcywh(boxes_v) if boxes_v.numel() > 0 else boxes_v
        confs_v = confs[valid]
        qidx_v = query_indices[valid]
        kp2ds_v = kp2ds[valid]
        querys = querys[0]
        querys_v = querys[valid]

        # posetrack: 0.7
        # 3dpw: 0.99
        if self.nms_threshold != 1.0:
            keep_nms = nms_xyxy(boxes_v, confs_v, iou_thresh=self.nms_threshold)

            boxes_v = boxes_v[keep_nms]
            # Original code:
            # confs_v = confs_v[keep_nms]
            # qidx_v = qidx_v[keep_nms]
            # kp2ds_v = kp2ds_v[keep_nms]
            # querys_v = querys_v[keep_nms]
            
            # Added for Kalman Filter
            boxes_v_cxcywh = boxes_v_cxcywh[keep_nms] if boxes_v_cxcywh.numel() > 0 else boxes_v_cxcywh
            confs_v = confs_v[keep_nms]
            qidx_v = qidx_v[keep_nms]
            kp2ds_v = kp2ds_v[keep_nms]
            querys_v = querys_v[keep_nms]

        # --- 2) If no tracks yet, initialize from current frame ---
        if len(self.tracks) == 0:
            # _ = self.init_frame(boxes_v, confs_v, qidx_v, kps=kp2ds_v, gt_boxes=gt_boxes, querys=querys_v, nms_iou=0.3)
            # posetrack
            _ = self.init_frame(boxes_v, confs_v, qidx_v, kps=kp2ds_v, gt_boxes=gt_boxes, querys=querys_v, nms_iou=0.7)
            # Ensure new tracks start with prev_box = box
            for tid in self.tracks:
                if "prev_box" not in self.tracks[tid]:
                    self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
                if "prev_kps" not in self.tracks[tid]:
                    self.tracks[tid]["prev_kps"] = self.tracks[tid]["kps"].clone()
                if "prev_query" not in self.tracks[tid]:
                    self.tracks[tid]["prev_query"] = self.tracks[tid]["query"].clone()
                    self.tracks[tid]["prev_query_list"] = [self.tracks[tid]["query"].clone()]

        else:
            track_ids = list(self.tracks.keys())

            if boxes_v.numel() > 0 and len(track_ids) > 0:
                # Original code (IoU against previous boxes):
                # prev_boxes_stack = torch.stack([self.tracks[tid]["box"] for tid in track_ids], dim=0)
                # iou_mat = box_iou_xyxy(prev_boxes_stack, boxes_v)
                
                # 1) Kalman Filter predict step for all tracks
                pred_boxes = []
                for tid in track_ids:
                    # if "kf" in self.tracks[tid]:
                    #     pred_xyxy = self.tracks[tid]["kf"].predict()
                    #     pred_boxes.append(pred_xyxy.unsqueeze(0))
                    # else:
                    #     # If KF doesn't exist (backward compatibility), use current box
                    pred_boxes.append(self.tracks[tid]["box"].unsqueeze(0))
                
                pred_boxes_stack = torch.cat(pred_boxes, dim=0) if pred_boxes else boxes_v.new_zeros((0, 4))
                
                # IoU against predicted boxes from Kalman Filter (shape: T x D)
                iou_mat = box_iou_xyxy(pred_boxes_stack, boxes_v)

                prev_querys_stack = torch.stack([self.tracks[tid]["query"] for tid in track_ids], dim=0)
                # prev_querys_stack = torch.stack([torch.mean(torch.stack(self.tracks[tid]["prev_query_list"]), 0) for tid in track_ids], dim=0)

                cos_sim = F.cosine_similarity(
                    prev_querys_stack.unsqueeze(1),  # [2,1,768]
                    querys_v.unsqueeze(0),      # [1,52,768]
                    dim=-1
                )  # [2,52]


                # prev_keypoints = torch.stack([self.tracks[tid]["kps"] for tid in track_ids], dim=0)
                # p = torch.tensor(prev_keypoints) / 1288
                # c = torch.ones_like(p[..., 0])
                # p2 = torch.tensor(kp2ds_v) / 1288
                # c2 = torch.ones_like(p2[..., 0])
                # oks_matrix = normalized_weighted_score(p, c, p2, c2)  # shape: [N, M]

                # Greedy matching in descending IoU
                # coords = torch.nonzero((iou_mat * 0.5 + oks_matrix * 2 * 0.5) >= self.iou_match_thresh, as_tuple=False)
                iou_mat = iou_mat * 0.1 + cos_sim * 0.9
                # iou_mat = cos_sim 
                # iou_mat = iou_mat
                # iou_mat = iou_mat * 0.1 + cos_sim * 0.9
                # iou_mat = iou_mat # * 0.1 + cos_sim * 0.9
                # iou_mat = iou_mat * 0.1 + oks_matrix * 0.6 + cos_sim * 0.3

                # iou_mat = iou_mat * 0.2 + oks_matrix * 0.4 + cos_sim * 0.4
                # coords = torch.nonzero((iou_mat) >= self.iou_match_thresh, as_tuple=False)
                coords = torch.nonzero((iou_mat) >= self.iou_match_thresh, as_tuple=False)
                used_tracks, matched_dets = set(), set()
                tracking_ids = []

                if coords.numel() > 0:
                    scores = iou_mat[coords[:, 0], coords[:, 1]]
                    order = torch.argsort(scores, descending=True)

                    for k in order.tolist():
                        t = coords[k, 0].item()
                        d = coords[k, 1].item()

                        if t in used_tracks or d in matched_dets:
                            continue

                        # if t in used_tracks:
                        #     continue

                        # if d in matched_dets:
                        #     if not self.is_add_new:
                        #         t_mask = (coords[:, 0] == t)
                        #         t_rows = torch.nonzero(t_mask, as_tuple=False).squeeze(1)
                        #         t_scores = scores[t_rows]
                        #         t_local_order = torch.argsort(t_scores, descending=True).tolist()

                        #         found = False
                        #         for r in [t_rows[i].item() for i in t_local_order]:
                        #             d2 = coords[r, 1].item()
                        #             if d2 == d or d2 in matched_dets:
                        #                 continue

                        #             tid = track_ids[t]
                        #             tracking_ids.append(tid)

                        #             self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
                        #             self.tracks[tid]["prev_kps"] = self.tracks[tid]["kps"].clone()
                        #             self.tracks[tid]["prev_query"] = self.tracks[tid]["query"].clone()
                        #             self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]

                        #             # Original code:
                        #             # self.tracks[tid]["box"] = boxes_v[d2].detach()
                                    
                        #             # Update Kalman Filter with measurement
                        #             # if "kf" in self.tracks[tid]:
                        #             #     meas = boxes_v_cxcywh[d2]
                        #             #     new_xyxy = self.tracks[tid]["kf"].update(meas)
                        #             #     self.tracks[tid]["box"] = new_xyxy.detach()
                        #             # else:
                        #             self.tracks[tid]["box"] = boxes_v[d2].detach()
                                    
                        #             self.tracks[tid]["kps"] = kp2ds_v[d2].detach()
                        #             self.tracks[tid]["query"] = querys_v[d2].detach()
                        #             self.tracks[tid]["prev_query_list"].append(querys_v[d2].detach())
                        #             self.tracks[tid]["prev_query_list"] = self.tracks[tid]["prev_query_list"][:self.prev_query_num]

                        #             self.tracks[tid]["age"] = 0
                        #             self.tracks[tid]["query_idx"] = int(qidx_v[d2])

                        #             used_tracks.add(t)
                        #             matched_dets.add(d2)
                        #             found = True
                        #             break

                        #         if not found:
                        #             continue  
                        #     else:
                        #         continue  

                        # else:
                        tid = track_ids[t]
                        tracking_ids.append(tid)

                        self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
                        self.tracks[tid]["prev_kps"] = self.tracks[tid]["kps"].clone()
                        self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]
                        self.tracks[tid]["prev_query_list"].append(querys_v[d].detach())
                        self.tracks[tid]["prev_query_list"] = self.tracks[tid]["prev_query_list"][:self.prev_query_num]

                        # Original code:
                        # self.tracks[tid]["box"] = boxes_v[d].detach()
                        
                        # Update Kalman Filter with measurement
                        # if "kf" in self.tracks[tid]:
                        #     meas = boxes_v_cxcywh[d]
                        #     new_xyxy = self.tracks[tid]["kf"].update(meas)
                        #     self.tracks[tid]["box"] = new_xyxy.detach()
                        # else:
                        self.tracks[tid]["box"] = boxes_v[d].detach()
                        
                        self.tracks[tid]["kps"] = kp2ds_v[d].detach()
                        self.tracks[tid]["query"] = querys_v[d].detach()
                        self.tracks[tid]["age"] = 0
                        self.tracks[tid]["query_idx"] = int(qidx_v[d])

                        used_tracks.add(t)
                        matched_dets.add(d)
            
                # Unmatched detections → candidates for new tracks (start with prev_box = box)
                all_dets = set(range(boxes_v.size(0)))
                unmatched = list(all_dets - matched_dets)

                # hasattr(self, 'is_add_new'), self.is_add_new)
                if len(unmatched) > 0 and (hasattr(self, 'is_add_new') and self.is_add_new):
                    um = torch.tensor(unmatched, device=boxes_v.device)

                    # prev_boxes = torch.stack(
                    #     [self.tracks[tid]["prev_box"] for tid in track_ids],
                    #     dim=0
                    # ) if len(track_ids) > 0 else boxes_xyxy_v.new_zeros((0,4))
                    # iou_mat = box_iou_xyxy(prev_boxes, boxes_v)

                    # # (A) Filter by max IoU vs *previous* boxes (as before)
                    # keep_mask_existing = []
                    # for d in um.tolist():
                    #     max_iou_prev = iou_mat[:, d].max().item() if iou_mat.numel() else 0.0
                    #     keep_mask_existing.append(
                    #         # Previous one
                    #         (max_iou_prev < self.iou_new_thresh) and (confs_v[d] >= self.conf_thresh+0.2)
                    #         # (max_iou_prev < self.iou_new_thresh) and (confs_v[d] >= self.conf_thresh+0.2)
                    #         # (max_iou_prev < self.iou_new_thresh) and (confs_v[d] >= self.conf_thresh+0.2)
                    #     )
                    # um = um[torch.tensor(keep_mask_existing, device=um.device, dtype=torch.bool)]

                    # (B) NEW: Filter by max IoU vs *current* boxes (after updates)
                    # This prevents spawning a new track that heavily overlaps an already-updated track box.
                    if um.numel() > 0:
                        active_curr_ids = list(self.tracks.keys())
                        curr_boxes_stack = (
                            torch.stack([self.tracks[tid]["box"] for tid in active_curr_ids], dim=0)
                            if len(active_curr_ids) > 0 else boxes_v.new_zeros((0, 4))
                        )
                        keep_mask_current = []
                        for d in um.tolist():
                            if curr_boxes_stack.numel() == 0:
                                keep_mask_current.append(True)
                                continue
                            ious_curr = box_iou_xyxy(curr_boxes_stack, boxes_v[d].unsqueeze(0)).squeeze(1)  # (N,)
                            max_iou_curr = float(ious_curr.max().item()) if ious_curr.numel() > 0 else 0.0
                            # keep_mask_current.append(max_iou_curr < self.iou_new_thresh and confs_v[d] >= self.conf_thresh)
                            keep_mask_current.append(max_iou_curr < self.iou_new_thresh and confs_v[d] >= self.conf_thresh+0.2)
                            # keep_mask_current.append(max_iou_curr < self.iou_new_thresh and confs_v[d] >= self.conf_thresh+0.15)
                            # if max_iou_curr < self.iou_new_thresh and confs_v[d] >= self.conf_thresh+0.2:
                            #     print( max_iou_curr, confs_v[d], self.conf_thresh+0.2, self.iou_new_thresh)
                            # keep_mask_current.append(max_iou_curr < self.iou_new_thresh and confs_v[d] >= self.conf_thresh)

                        um = um[torch.tensor(keep_mask_current, device=um.device, dtype=torch.bool)]

                    # if um.numel() > 0:
                    #     active_curr_ids = list(self.tracks.keys())
                    #     curr_boxes_stack = (
                    #         torch.stack([self.tracks[tid]["query"] for tid in active_curr_ids], dim=0)
                    #         if len(active_curr_ids) > 0 else querys_v.new_zeros((0, 768))
                    #     )
                    #     keep_mask_current = []
                    #     for d in um.tolist():
                    #         if curr_boxes_stack.numel() == 0:
                    #             keep_mask_current.append(True)
                    #             continue
                            
                    #         cos_sim = F.cosine_similarity(
                    #             curr_boxes_stack.unsqueeze(1),  # [2,1,768]
                    #             querys_v[d].unsqueeze(0),      # [1,52,768]
                    #             dim=-1
                    #         )  # [2,52]

                    #         # ious_curr = box_iou_xyxy(curr_boxes_stack, querys_v[d].unsqueeze(0)).squeeze(1)  # (N,)
                    #         max_iou_curr = float(ious_curr.max().item()) if ious_curr.numel() > 0 else 0.0
                    #         keep_mask_current.append(max_iou_curr < self.iou_new_thresh and confs_v[d] >= self.conf_thresh+0.2)

                    #     um = um[torch.tensor(keep_mask_current, device=um.device, dtype=torch.bool)]

                    # (B2) ReID pass: try to attach unmatched dets to recently-unmatched tracks
                    if um.numel() > 0:
                        # choose candidate tracks for reid: small age (e.g., 1..reid_max_age) and not matched this frame
                        reid_max_age = getattr(self, "reid_max_age", 3)
                        iou_reid_thresh = getattr(self, "iou_reid_thresh", 0.3)  # looser than iou_match_thresh
                        cand_tids = [tid for tid in self.tracks
                                    # if 1 <= self.tracks[tid]["age"] <= reid_max_age ]
                                    if 0 <= self.tracks[tid]["age"] <= reid_max_age and tid not in used_tracks]

                        if len(cand_tids) > 0:
                            # Original code:
                            cand_prev = torch.stack([self.tracks[tid]["box"] for tid in cand_tids], dim=0)  # use last box or a predicted box
                            # # (선택) 칼만/상보필터 등으로 cand_prev를 예측 박스로 바꾸면 더 견고
                            
                            # # Use Kalman Filter predicted boxes for ReID
                            # cand_pred_boxes = []
                            # for tid in cand_tids:
                            #     if "kf" in self.tracks[tid]:
                            #         # Already predicted in the main matching loop, but we need to get the predicted box
                            #         # If predict was called, we can use it again or store it
                            #         pred_xyxy = self.tracks[tid]["kf"].get_xyxy()
                            #         cand_pred_boxes.append(pred_xyxy.unsqueeze(0))
                            #     else:
                            #         cand_pred_boxes.append(self.tracks[tid]["box"].unsqueeze(0))
                            # cand_prev = torch.cat(cand_pred_boxes, dim=0) if cand_pred_boxes else boxes_v.new_zeros((0, 4))
                            det_boxes = boxes_v[um]  # unmatched dets

                            iou_reid = box_iou_xyxy(cand_prev, det_boxes)  # (Tc, Ud)
                            coords = torch.nonzero(iou_reid >= iou_reid_thresh, as_tuple=False)
                            if coords.numel() > 0:
                                scores = iou_reid[coords[:,0], coords[:,1]]
                                order = torch.argsort(scores, descending=True)
                                used_cand, used_det = set(), set()
                                newly_attached = []
                                for k in order.tolist():
                                    ti = coords[k,0].item()
                                    di = coords[k,1].item()
                                    if ti in used_cand or di in used_det:
                                        continue
                                    tid = cand_tids[ti]
                                    d_global = um[di].item()
                                    tracking_ids.append(tid)

                                    # attach: mimic a normal match update
                                    self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
                                    self.tracks[tid]["prev_kps"] = self.tracks[tid]["kps"].clone()
                                    self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]
                                    self.tracks[tid]["prev_query_list"].append(querys_v[d_global].detach())
                                    self.tracks[tid]["prev_query_list"] = self.tracks[tid]["prev_query_list"][:self.prev_query_num]
                                    
                                    # Original code:
                                    # self.tracks[tid]["box"] = boxes_v[d_global].detach()
                                    
                                    # Update Kalman Filter with measurement
                                    # if "kf" in self.tracks[tid]:
                                    #     meas = boxes_v_cxcywh[d_global]
                                    #     new_xyxy = self.tracks[tid]["kf"].update(meas)
                                    #     self.tracks[tid]["box"] = new_xyxy.detach()
                                    # else:
                                    self.tracks[tid]["box"] = boxes_v[d_global].detach()
                                    
                                    self.tracks[tid]["kps"] = kp2ds_v[d_global].detach()
                                    self.tracks[tid]["query"] = querys_v[d_global].detach()
                                    self.tracks[tid]["age"] = 0
                                    self.tracks[tid]["query_idx"] = int(qidx_v[d_global])

                                    used_cand.add(ti)
                                    used_det.add(di)
                                    newly_attached.append(di)

                                # remove attached dets from um
                                if newly_attached:
                                    keep_mask = torch.ones(um.numel(), dtype=torch.bool, device=um.device)
                                    keep_mask[torch.tensor(newly_attached, device=um.device, dtype=torch.long)] = False
                                    um = um[keep_mask]

                    if um.numel() > 0:
                        # (C) Per-origin de-dup: keep the highest-confidence per origin_qidx
                        best_per_origin = {}
                        for d in um.tolist():
                            o = int(qidx_v[d])
                            if o not in best_per_origin or confs_v[d] > confs_v[best_per_origin[o]]:
                                best_per_origin[o] = d
                        per_origin_idxs = torch.tensor(list(best_per_origin.values()), device=boxes_v.device)

                        # (D) NMS among remaining detections to collapse spatial duplicates
                        u_boxes = boxes_v[per_origin_idxs]
                        u_confs = confs_v[per_origin_idxs]
                        keep_nms = nms_xyxy(u_boxes, u_confs, iou_thresh=self.nms_threshold)
                        final_new = per_origin_idxs[keep_nms].tolist()

                        # (E) Avoid spawning a duplicate active track with the same origin
                        active_by_origin = {
                            self.tracks[tid]["origin_qidx"]: tid
                            for tid in self.tracks
                            if "origin_qidx" in self.tracks[tid]
                        }

                        for d in final_new:
                            org = int(qidx_v[d])
                            if org in active_by_origin:
                                continue

                            tid_new = self.next_id
                            self.next_id += 1
                            
                            # Original code:
                            # self.tracks[tid_new] = {
                            #     "box": boxes_v[d].detach(),
                            #     "prev_box": boxes_v[d].detach(),  # initialize prev=curr
                            #     "age": 0,
                            #     "query_idx": org,
                            #     "origin_qidx": org,
                            # }
                            
                            # Initialize Kalman Filter for new track
                            init_box = boxes_v[d]
                            kf = TorchKalman(dt=self.dt, device=init_box.device)
                            kf.init_from_xyxy(init_box)
                            
                            self.tracks[tid_new] = {
                                "box": init_box.detach(),
                                "prev_box": init_box.detach().clone(),  # initialize prev=curr
                                "age": 0,
                                "query_idx": org,
                                "origin_qidx": org,
                                "kf": kf,
                                "kps": kp2ds_v[d].detach(),
                                "query": querys_v[d].detach(),
                                "prev_query_list": [querys_v[d].detach().clone()],
                            }
            # import pdb; pdb.set_trae()
        # if self.frame_id > 107:
        #     import numpy as np
        #     tdets = np.array(list(used_tracks))
        #     _track_ids = np.array(track_ids)
        #     # track_ids
        #     import pdb; pdb.set_trace()
            

        # --- 3) Resolve track–track conflicts: within any IoU>=overlap_iou group keep only the most self-consistent track ---
        if self.is_add_new:
            active_ids = list(self.tracks.keys())
            if len(active_ids) >= 2:
                cur_boxes = torch.stack([self.tracks[tid]["box"] for tid in active_ids], dim=0)
                iou_tt = box_iou_xyxy(cur_boxes, cur_boxes)  # (N, N)
                N = iou_tt.size(0)

                # Connected components over edges (iou >= overlap_iou)
                visited = [False] * N
                groups = []
                for i in range(N):
                    if visited[i]:
                        continue
                    stack = [i]
                    comp = []
                    visited[i] = True
                    while stack:
                        u = stack.pop()
                        comp.append(u)
                        nbrs = (iou_tt[u] >= overlap_iou).nonzero(as_tuple=False).flatten().tolist()
                        for v in nbrs:
                            if not visited[v]:
                                visited[v] = True
                                stack.append(v)
                    groups.append(comp)

                # In each group, keep the track with max IoU(prev_box, curr_box); rollback others for this frame
                for comp in groups:
                    if len(comp) <= 1:
                        continue

                    self_ious = []
                    for idx in comp:
                        tid = active_ids[idx]
                        curr = self.tracks[tid]["box"].unsqueeze(0)

                        prevb = self.tracks[tid].get("prev_box", None)
                        if prevb is None:
                            pb = prev_box.get(tid, None)
                            if pb is None:
                                self_ious.append(-1.0)  # penalize if we cannot compute consistency
                                continue
                            prevb = pb

                        prevb = prevb.unsqueeze(0)
                        val = box_iou_xyxy(prevb, curr)[0, 0].item()
                        self_ious.append(val)

                    winner_local_idx = int(torch.tensor(self_ious).argmax().item())
                    winner_tid = active_ids[comp[winner_local_idx]]

                    # Rollback losers to their pre-update state; age = prev_age + 1
                    for idx in comp:
                        tid = active_ids[idx]
                        if tid == winner_tid:
                            continue
                        if tid in prev_box:
                            self.tracks[tid]["box"] = prev_box[tid]
                        if tid in prev_qid:
                            self.tracks[tid]["query_idx"] = prev_qid[tid]
                        self.tracks[tid]["age"] = prev_age.get(tid, self.tracks[tid]["age"]) + 1

        # --- 4) Drop stale tracks ---
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        # --- 5) Return active matched ids and index maps (age == 0 means matched this frame) ---
        active_matched_ids = [tid for tid in self.tracks.keys() if self.tracks[tid]["age"] == 0]
        id2qidx = {tid: self.tracks[tid]["query_idx"] for tid in active_matched_ids}
        id2origin = {
            tid: self.tracks[tid].get("origin_qidx", self.tracks[tid]["query_idx"])
            for tid in active_matched_ids
        }
        self.active_ids = active_matched_ids
        # print(self.active_ids, self.next_id)
        return active_matched_ids, id2qidx, id2origin

# class QueryTracker:
#     def __init__(self,
#                  conf_thresh=0.5,
#                  iou_match_thresh=0.5,
#                  iou_new_thresh=0.3,
#                  max_age=15,
#                  boxes_format="xyxy",
#                  pr_conf_thresh=0.9,
#                  is_add_new=True,
#                  is_size_filter=True,
#                  nms_threshold=0.7,
#                  dt=1.0):
#         self.conf_thresh = conf_thresh
#         self.iou_match_thresh = iou_match_thresh
#         self.iou_new_thresh = iou_new_thresh
#         self.max_age = max_age
#         assert boxes_format in ("xyxy", "cxcywh")
#         self.boxes_format = boxes_format
#         self.pr_conf_thresh = pr_conf_thresh

#         self.dt = dt
#         self.next_id = 0
#         self.tracks = {}   # tid -> dict(box, age, query_idx, origin_qidx, kf, prev_box, prev_query_idx)

#     def reset(self):
#         self.next_id = 0
#         self.tracks = {}

#     def _to_xyxy(self, boxes):
#         if boxes.numel() == 0:
#             return boxes
#         return box_cxcywh_to_xyxy(boxes)

#     def _to_cxcywh(self, boxes):
#         if boxes.numel() == 0:
#             return boxes
#         return box_xyxy_to_cxcywh(boxes)

#     @torch.no_grad()
#     def init_frame(self, boxes, confs, query_indices, nms_iou=0.3):
#         """Initialize tracks from this frame’s valid detections (no KF predict step)."""
#         if boxes.numel() == 0:
#             return []

#         boxes_xyxy = self._to_xyxy(boxes)
#         keep = nms_xyxy(boxes_xyxy, confs, iou_thresh=nms_iou)
#         if keep.numel() == 0:
#             return []

#         keep_boxes_xyxy = boxes_xyxy[keep]
#         keep_confs = confs[keep]
#         keep_qidx  = query_indices[keep]

#         active_ids = []
#         device = keep_boxes_xyxy.device
#         for bxyxy, c, q in zip(keep_boxes_xyxy, keep_confs, keep_qidx):
#             if c < self.conf_thresh:
#                 continue
#             tid = self.next_id
#             self.next_id += 1

#             kf = TorchKalman(dt=self.dt, device=device)
#             kf.init_from_xyxy(bxyxy)

#             self.tracks[tid] = {
#                 "box": bxyxy.detach().clone(),        # xyxy
#                 "pred_box": bxyxy.detach().clone(),
#                 "age": 0,
#                 "query_idx": int(q),
#                 "origin_qidx": int(q),
#                 "kf": kf,
#                 "prev_box": bxyxy.detach().clone(),
#                 "prev_query_idx": int(q),
#             }
#             active_ids.append(tid)
#         return active_ids

#     @torch.no_grad()
#     def update(self, boxes, confs, query_indices, overlap_iou=0.8):
#         """
#         KF-강화 업데이트:
#           1) 모든 트랙 KF.predict() → 예측박스로 매칭(IoU)
#           2) 매칭된 트랙 KF.update(z=cxcywh)
#           3) unmatched det → 새 트랙 생성
#           4) 트랙-트랙 충돌 해결 & stale drop
#         """
#         # 0) 사전 준비
#         boxes_xyxy = self._to_xyxy(boxes)
#         boxes_cxcywh = boxes

#         prev_box, prev_age, prev_qid = {}, {}, {}
#         for tid in list(self.tracks.keys()):
#             prev_box[tid] = self.tracks[tid]["box"].clone()
#             prev_age[tid] = self.tracks[tid]["age"]
#             prev_qid[tid] = self.tracks[tid]["query_idx"]
#             self.tracks[tid]["age"] += 1  # 기본 age++

#         widths  = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
#         heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
#         min_w, min_h = 50, 50
#         size_valid = (widths >= min_w) & (heights >= min_h)

#         valid = (confs >= self.conf_thresh) & size_valid
#         valid[:self.next_id] = False
#         pr_valid = (confs >= self.pr_conf_thresh) & size_valid
#         pr_valid[self.next_id:] = False
#         valid = pr_valid | valid
#         boxes_v_xyxy   = boxes_xyxy[valid]
#         boxes_v_cxcywh = boxes_cxcywh[valid]
#         confs_v        = confs[valid]
#         qidx_v         = query_indices[valid]

#         # 1) 트랙 예측 (KF) → 매칭용 예측 박스
#         track_ids = list(self.tracks.keys())
#         pred_boxes = []
#         for tid in track_ids:
#             kf = self.tracks[tid]["kf"]
#             pred_xyxy = kf.predict()                     # (4,)
#             self.tracks[tid]["pred_box"] = pred_xyxy.detach().clone() 
#             pred_boxes.append(pred_xyxy.unsqueeze(0))
#         pred_boxes = torch.cat(pred_boxes, dim=0) if pred_boxes else boxes_v_xyxy.new_zeros((0,4))

#         # 2) 매칭 (예측 vs 현재 valid det)
#         used_tracks, matched_dets = set(), set()
#         if pred_boxes.numel() > 0 and boxes_v_xyxy.numel() > 0:
#             iou_mat = box_iou_xyxy(pred_boxes, boxes_v_xyxy)
#             coords = torch.nonzero(iou_mat >= self.iou_match_thresh, as_tuple=False)

#             if coords.numel() > 0:
#                 scores = iou_mat[coords[:, 0], coords[:, 1]]
#                 order = torch.argsort(scores, descending=True)
#                 for k in order.tolist():
#                     tr_i = coords[k, 0].item()
#                     det_j = coords[k, 1].item()
#                     if tr_i in used_tracks or det_j in matched_dets:
#                         continue
#                     tid = track_ids[tr_i]

#                     # rollback anchor
#                     self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
#                     self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]

#                     # 2-a) KF.update with measurement (cxcywh)
#                     meas = boxes_v_cxcywh[det_j]
#                     new_xyxy = self.tracks[tid]["kf"].update(meas)

#                     # 상태 반영
#                     self.tracks[tid]["box"] = new_xyxy.detach().clone()
#                     self.tracks[tid]["age"] = 0
#                     self.tracks[tid]["query_idx"] = int(qidx_v[det_j])

#                     used_tracks.add(tr_i)
#                     matched_dets.add(det_j)
#         else:
#             iou_mat = boxes_v_xyxy.new_zeros((pred_boxes.size(0), boxes_v_xyxy.size(0)))

#         # 3) unmatched det → 새 트랙 생성 (KF 초기화)
#         all_dets = set(range(boxes_v_xyxy.size(0)))
#         unmatched = list(all_dets - matched_dets)

#         # (A) 기존 트랙들과의 최대 IoU가 너무 크면 생성 억제
#         new_idxs = []
#         if len(unmatched) > 0:
#             for j in unmatched:
#                 max_iou_prev = iou_mat[:, j].max().item() if iou_mat.numel() else 0.0
#                 if (max_iou_prev < self.iou_new_thresh) and (confs_v[j] >= self.conf_thresh):
#                     new_idxs.append(j)

#         # (B) (선택) 현재 활성 박스와의 과도 중복 억제
#         if new_idxs:
#             active_xyxy = torch.stack([self.tracks[t]["box"] for t in track_ids], dim=0) if track_ids else boxes_v_xyxy.new_zeros((0,4))
#             filtered = []
#             for j in new_idxs:
#                 if active_xyxy.numel() == 0:
#                     filtered.append(j); continue
#                 ious = box_iou_xyxy(active_xyxy, boxes_v_xyxy[j:j+1]).squeeze(1)
#                 if float(ious.max().item()) < self.iou_new_thresh:
#                     filtered.append(j)
#             new_idxs = filtered

#         if len(unmatched) > 0 and len(track_ids) > 0:
#             reid_max_age   = getattr(self, "reid_max_age", 5)
#             iou_reid_thresh = getattr(self, "iou_reid_thresh", 0.3)

#             # candidate tracks: not matched in first pass, and 1 <= age <= reid_max_age
#             used_track_indices = set(used_tracks)  # indices in track_ids already used in first pass
#             cand_t_indices = []
#             cand_pred_boxes = []
#             for tr_i, tid in enumerate(track_ids):
#                 if tr_i in used_track_indices:
#                     continue
#                 age = self.tracks[tid]["age"]
#                 if 1 <= age <= reid_max_age:
#                     cand_t_indices.append(tr_i)
#                     # use KF predicted box saved earlier
#                     cand_pred_boxes.append(self.tracks[tid]["pred_box"].unsqueeze(0))

#             if cand_t_indices:
#                 cand_pred_boxes = torch.cat(cand_pred_boxes, dim=0)  # (Tc,4)
#                 det_boxes_um = boxes_v_xyxy[unmatched]               # (Ud,4)

#                 iou_reid = box_iou_xyxy(cand_pred_boxes, det_boxes_um)  # (Tc, Ud)
#                 coords = torch.nonzero(iou_reid >= iou_reid_thresh, as_tuple=False)

#                 if coords.numel() > 0:
#                     scores = iou_reid[coords[:,0], coords[:,1]]
#                     order = torch.argsort(scores, descending=True)

#                     cand_used, det_used = set(), set()
#                     for k in order.tolist():
#                         ci = coords[k,0].item()  # index in cand_t_indices
#                         di = coords[k,1].item()  # index in 'unmatched' list
#                         if ci in cand_used or di in det_used:
#                             continue

#                         tr_i = cand_t_indices[ci]       # index in track_ids
#                         tid  = track_ids[tr_i]
#                         det_j_global = unmatched[di]     # index in boxes_v_xyxy / boxes_v_cxcywh

#                         # rollback anchor before applying
#                         self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
#                         self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]

#                         # KF.update with measurement (cx,cy,w,h)
#                         meas = boxes_v_cxcywh[det_j_global]
#                         new_xyxy = self.tracks[tid]["kf"].update(meas)

#                         # commit
#                         self.tracks[tid]["box"] = new_xyxy.detach().clone()
#                         self.tracks[tid]["age"] = 0
#                         self.tracks[tid]["query_idx"] = int(qidx_v[det_j_global])

#                         # also mark as used so they won't spawn new tracks later
#                         used_tracks.add(tr_i)
#                         matched_dets.add(det_j_global)
#                         cand_used.add(ci)
#                         det_used.add(di)

#                     # recompute unmatched after reattach
#                     all_dets = set(range(boxes_v_xyxy.size(0)))
#                     unmatched = list(all_dets - matched_dets)

#         # (C) 생성
#         for j in new_idxs:
#             tid_new = self.next_id
#             self.next_id += 1
#             init_xyxy = boxes_v_xyxy[j]
#             kf = TorchKalman(dt=self.dt, device=init_xyxy.device)
#             kf.init_from_xyxy(init_xyxy)
#             self.tracks[tid_new] = {
#                 "box": init_xyxy.detach().clone(),
#                 "prev_box": init_xyxy.detach().clone(),
#                 "age": 0,
#                 "query_idx": int(qidx_v[j]),
#                 "origin_qidx": int(qidx_v[j]),
#                 "kf": kf,
#                 "prev_query_idx": int(qidx_v[j]),
#             }

#         # 4) 트랙-트랙 충돌 해결 (네 기존 로직 유지: self-consistency IoU winner keep)
#         active_ids = list(self.tracks.keys())
#         if len(active_ids) >= 2:
#             cur_boxes = torch.stack([self.tracks[tid]["box"] for tid in active_ids], dim=0)
#             iou_tt = box_iou_xyxy(cur_boxes, cur_boxes)
#             N = iou_tt.size(0)
#             visited = [False]*N
#             groups = []
#             for i in range(N):
#                 if visited[i]: continue
#                 stack=[i]; comp=[]; visited[i]=True
#                 while stack:
#                     u = stack.pop()
#                     comp.append(u)
#                     nbrs = (iou_tt[u] >= overlap_iou).nonzero(as_tuple=False).flatten().tolist()
#                     for v in nbrs:
#                         if not visited[v]:
#                             visited[v]=True; stack.append(v)
#                 groups.append(comp)

#             for comp in groups:
#                 if len(comp) <= 1: continue
#                 self_ious=[]
#                 for idx in comp:
#                     tid = active_ids[idx]
#                     curr = self.tracks[tid]["box"].unsqueeze(0)
#                     prevb = self.tracks[tid].get("prev_box", None)
#                     if prevb is None: prevb = prev_box.get(tid, None)
#                     if prevb is None:
#                         self_ious.append(-1.0); continue
#                     val = box_iou_xyxy(prevb.unsqueeze(0), curr)[0,0].item()
#                     self_ious.append(val)
#                 winner_local = int(torch.tensor(self_ious).argmax().item())
#                 winner_tid = active_ids[comp[winner_local]]
#                 for idx in comp:
#                     tid = active_ids[idx]
#                     if tid == winner_tid: continue
#                     if tid in prev_box:
#                         self.tracks[tid]["box"] = prev_box[tid]
#                     if tid in prev_qid:
#                         self.tracks[tid]["query_idx"] = prev_qid[tid]
#                     self.tracks[tid]["age"] = prev_age.get(tid, self.tracks[tid]["age"]) + 1

#         # 5) 오래된 트랙 drop
#         for tid in list(self.tracks.keys()):
#             if self.tracks[tid]["age"] > self.max_age:
#                 del self.tracks[tid]

#         # 6) 이번 프레임에 매칭된(active) ID만 반환
#         active_matched_ids = [tid for tid in self.tracks if self.tracks[tid]["age"] == 0]
#         id2qidx = {tid: self.tracks[tid]["query_idx"] for tid in active_matched_ids}
#         id2origin = {tid: self.tracks[tid].get("origin_qidx", self.tracks[tid]["query_idx"])
#                      for tid in active_matched_ids}
#         self.active_ids = active_matched_ids
#         return active_matched_ids, id2qidx, id2origin

    def select_tracks_by_self_iou(self, overlap_iou=0.8):
        """
        Selects the best track per overlapping group based on self IoU.

        - Step 1: Compute IoU between all current track boxes.
        - Step 2: Group tracks that overlap (IoU >= overlap_iou).
        - Step 3: For each group, compute self IoU = IoU(prev_box, curr_box)
                  and select the track with the highest self IoU.
        - Step 4: Return the list of selected track IDs.
        """
        tids = list(self.tracks.keys())
        if len(tids) == 0:
            return []

        # current boxes
        cur_boxes = torch.stack([self.tracks[tid]["box"].detach().cpu() for tid in tids], dim=0)
        iou_tt = box_iou_xyxy(cur_boxes, cur_boxes)  # N x N

        # Build overlap groups (connected components)
        N = len(tids)
        visited = [False] * N
        groups = []
        for i in range(N):
            if visited[i]:
                continue
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = (iou_tt[u] >= overlap_iou).nonzero(as_tuple=False).flatten().tolist()
                for v in nbrs:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            groups.append(comp)

        selected_tids = []
        for comp in groups:
            if len(comp) == 1:
                selected_tids.append(tids[comp[0]])
                continue

            # Compute self IoU for each track in the group
            self_iou_scores = []
            for idx in comp:
                tid = tids[idx]
                prev_box = self.tracks[tid].get("prev_box", None)
                if prev_box is None:
                    self_iou_scores.append(0.0)
                else:
                    pb = prev_box.unsqueeze(0).cpu()
                    cb = self.tracks[tid]["box"].unsqueeze(0).cpu()
                    val = box_iou_xyxy(pb, cb)[0, 0].item()
                    self_iou_scores.append(val)

            # Keep only the winner with highest self IoU
            best_idx = int(torch.tensor(self_iou_scores).argmax().item())
            winner_tid = tids[comp[best_idx]]
            selected_tids.append(winner_tid)

        return selected_tids