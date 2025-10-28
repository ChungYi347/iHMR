import torch 
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, nms_xyxy, box_iou_xyxy

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

class QueryTracker:
    def __init__(self,
                 conf_thresh=0.5,
                 iou_match_thresh=0.5,
                 iou_new_thresh=0.5,
                 max_age=15,
                 boxes_format="xyxy",
                 pr_conf_thresh=0.9):
        self.conf_thresh = conf_thresh
        self.iou_match_thresh = iou_match_thresh
        self.iou_new_thresh = iou_new_thresh
        self.max_age = max_age
        assert boxes_format in ("xyxy", "cxcywh")
        self.boxes_format = boxes_format
        self.pr_conf_thresh=pr_conf_thresh

        self.next_id = 0
        self.tracks = {}

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
    def init_frame(self, boxes, confs, query_indices, nms_iou=0.3):
        if boxes.numel() == 0:
            return []

        boxes_xyxy = boxes
        valid = confs > self.conf_thresh + 0.2
        boxes_xyxy = boxes_xyxy[valid]
        confs = confs[valid]
        query_indices  = query_indices[valid]

        keep = nms_xyxy(boxes_xyxy, confs, iou_thresh=nms_iou)
        if keep.numel() == 0:
            return []

        keep_boxes = boxes_xyxy[keep]
        keep_confs = confs[keep]
        keep_qidx  = query_indices[keep]

        active_ids = []
        for b, c, q in zip(keep_boxes, keep_confs, keep_qidx):
            if c < self.conf_thresh:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": b.detach(),
                "age": 0,
                "query_idx": int(q),   
                "origin_qidx": int(q)   
            }
            active_ids.append(tid)
        return active_ids

    # @torch.no_grad()
    # def update(self, boxes, confs, query_indices):
    #     boxes_xyxy = self._to_xyxy(boxes)

    #     for tid in list(self.tracks.keys()):
    #         self.tracks[tid]["age"] += 1

    #     valid = confs >= self.conf_thresh
    #     boxes_xyxy_v = boxes_xyxy[valid]
    #     confs_v = confs[valid]
    #     qidx_v = query_indices[valid]

    #     if len(self.tracks) == 0:
    #         _ = self.init_frame(boxes_xyxy_v, confs_v, qidx_v, nms_iou=0.6)
    #     else:
    #         track_ids = list(self.tracks.keys())
    #         if len(track_ids) == 0:
    #             # nothing to match
    #             pass
    #         else:
    #             prev_boxes = torch.stack(
    #                 [self.tracks[tid]["box"] for tid in track_ids],
    #                 dim=0
    #             ) if len(track_ids) > 0 else boxes_xyxy_v.new_zeros((0,4))
    #             if prev_boxes.numel() > 0 and boxes_xyxy_v.numel() > 0:
    #                 iou_mat = box_iou_xyxy(prev_boxes, boxes_xyxy_v)
    #                 coords = torch.nonzero(iou_mat >= self.iou_match_thresh, as_tuple=False)
    #                 if coords.numel() > 0:
    #                     pair_scores = iou_mat[coords[:,0], coords[:,1]]
    #                     order = torch.argsort(pair_scores, descending=True)
    #                     used_tracks = set()
    #                     matched_dets = set()
    #                     for k in order.tolist():
    #                         tr_i = coords[k,0].item()
    #                         det_j = coords[k,1].item()
    #                         if tr_i in used_tracks or det_j in matched_dets:
    #                             continue
    #                         tid = track_ids[tr_i]
    #                         self.tracks[tid]["box"] = boxes_xyxy_v[det_j].detach()
    #                         self.tracks[tid]["age"] = 0
    #                         self.tracks[tid]["query_idx"] = int(qidx_v[det_j])
    #                         used_tracks.add(tr_i)
    #                         matched_dets.add(det_j)

    #                     all_dets = set(range(boxes_xyxy_v.size(0)))
    #                     unmatched = list(all_dets - matched_dets)
    #                     for j in unmatched:
    #                         max_iou = iou_mat[:, j].max().item() if iou_mat.numel() else 0.0
    #                         if max_iou < self.iou_new_thresh and confs_v[j] >= self.conf_thresh:
    #                             tid = self.next_id
    #                             self.next_id += 1
    #                             self.tracks[tid] = {
    #                                 "box": boxes_xyxy_v[j].detach(),
    #                                 "age": 0,
    #                                 "query_idx": int(qidx_v[j]),
    #                                 "origin_qidx": int(qidx_v[j])
    #                             }
    #             else:
    #                 pass

    #     for tid in list(self.tracks.keys()):
    #         if self.tracks[tid]["age"] > self.max_age:
    #             del self.tracks[tid]

    #     active_ids = list(self.tracks.keys())
    #     id2qidx = {tid: self.tracks[tid]["query_idx"] for tid in active_ids}
    #     id2origin = {tid: self.tracks[tid]["origin_qidx"] for tid in active_ids}
    #     return active_ids, id2qidx, id2origin

    @torch.no_grad()
    def update(self, boxes, confs, query_indices, overlap_iou=0.8):
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
        for tid in list(self.tracks.keys()):
            prev_box[tid] = self.tracks[tid]["box"].clone()
            prev_age[tid] = self.tracks[tid]["age"]
            prev_qid[tid] = self.tracks[tid]["query_idx"]
            # default +1; will reset to 0 if matched
            self.tracks[tid]["age"] += 1

        widths  = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        min_w, min_h = 50, 50  

        # valid = confs >= self.conf_thresh
        # valid_idx = torch.nonzero(valid, as_tuple=True)[0]
        # valid_idx = torch.where(valid)[0]
        
        size_valid = (widths >= min_w) & (heights >= min_h)
        # valid = valid & size_valid

        valid = (confs >= self.conf_thresh) & size_valid
        valid[:self.next_id] = False
        valid = (confs >= self.conf_thresh) & size_valid
        valid[:self.next_id] = False
        pr_valid = (confs >= self.pr_conf_thresh) & size_valid
        pr_valid[self.next_id:] = False
        valid = pr_valid | valid
        boxes_v_xyxy   = boxes_xyxy[valid]
        # boxes_v_cxcywh = boxes_cxcywh[valid]
        confs_v        = confs[valid]
        qidx_v         = query_indices[valid]

        # --- 1) Keep only valid detections by confidence ---
        boxes_v = boxes_xyxy[valid]
        confs_v = confs[valid]
        qidx_v = query_indices[valid]

        keep_nms = nms_xyxy(boxes_v, confs_v, iou_thresh=0.7)

        boxes_v = boxes_v[keep_nms]
        confs_v = confs_v[keep_nms]
        qidx_v = qidx_v[keep_nms]

        # --- 2) If no tracks yet, initialize from current frame ---
        if len(self.tracks) == 0:
            _ = self.init_frame(boxes_v, confs_v, qidx_v, nms_iou=0.3)
            # Ensure new tracks start with prev_box = box
            for tid in self.tracks:
                if "prev_box" not in self.tracks[tid]:
                    self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()

        else:
            track_ids = list(self.tracks.keys())

            if boxes_v.numel() > 0 and len(track_ids) > 0:
                # IoU against previous boxes (shape: T x D)
                prev_boxes_stack = torch.stack([self.tracks[tid]["box"] for tid in track_ids], dim=0)
                iou_mat = box_iou_xyxy(prev_boxes_stack, boxes_v)

                # Greedy matching in descending IoU
                coords = torch.nonzero(iou_mat >= self.iou_match_thresh, as_tuple=False)
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

                        tid = track_ids[t]
                        tracking_ids.append(tid)

                        # Save rollback anchors just before applying the match
                        self.tracks[tid]["prev_box"] = self.tracks[tid]["box"].clone()
                        self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]

                        # Apply the match
                        self.tracks[tid]["box"] = boxes_v[d].detach()
                        self.tracks[tid]["age"] = 0
                        self.tracks[tid]["query_idx"] = int(qidx_v[d])

                        used_tracks.add(t)
                        matched_dets.add(d)
                
                # Unmatched detections → candidates for new tracks (start with prev_box = box)
                all_dets = set(range(boxes_v.size(0)))
                unmatched = list(all_dets - matched_dets)

                if len(unmatched) > 0:
                    um = torch.tensor(unmatched, device=boxes_v.device)

                    # (A) Filter by max IoU vs *previous* boxes (as before)
                    keep_mask_existing = []
                    for d in um.tolist():
                        max_iou_prev = iou_mat[:, d].max().item() if iou_mat.numel() else 0.0
                        keep_mask_existing.append(
                            (max_iou_prev < self.iou_new_thresh) and (confs_v[d] >= self.conf_thresh+0.2)
                        )
                    um = um[torch.tensor(keep_mask_existing, device=um.device, dtype=torch.bool)]

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
                            keep_mask_current.append(max_iou_curr < self.iou_new_thresh)

                        um = um[torch.tensor(keep_mask_current, device=um.device, dtype=torch.bool)]

                    # (B2) ReID pass: try to attach unmatched dets to recently-unmatched tracks
                    if um.numel() > 0:
                        # choose candidate tracks for reid: small age (e.g., 1..reid_max_age) and not matched this frame
                        reid_max_age = getattr(self, "reid_max_age", 3)
                        iou_reid_thresh = getattr(self, "iou_reid_thresh", 0.3)  # looser than iou_match_thresh
                        cand_tids = [tid for tid in self.tracks
                                    # if 1 <= self.tracks[tid]["age"] <= reid_max_age ]
                                    if 0 <= self.tracks[tid]["age"] <= reid_max_age and tid not in used_tracks]

                        if len(cand_tids) > 0:
                            cand_prev = torch.stack([self.tracks[tid]["box"] for tid in cand_tids], dim=0)  # use last box or a predicted box
                            # (선택) 칼만/상보필터 등으로 cand_prev를 예측 박스로 바꾸면 더 견고
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
                                    self.tracks[tid]["prev_query_idx"] = self.tracks[tid]["query_idx"]
                                    self.tracks[tid]["box"] = boxes_v[d_global].detach()
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
                        keep_nms = nms_xyxy(u_boxes, u_confs, iou_thresh=0.7)
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
                            self.tracks[tid_new] = {
                                "box": boxes_v[d].detach(),
                                "prev_box": boxes_v[d].detach(),  # initialize prev=curr
                                "age": 0,
                                "query_idx": org,
                                "origin_qidx": org,
                            }
        # if self.frame_id > 107:
        #     import numpy as np
        #     tdets = np.array(list(used_tracks))
        #     _track_ids = np.array(track_ids)
        #     # track_ids
        #     import pdb; pdb.set_trace()
            

        # --- 3) Resolve track–track conflicts: within any IoU>=overlap_iou group keep only the most self-consistent track ---
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
#                  pr_conf_thresh=0.7,
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