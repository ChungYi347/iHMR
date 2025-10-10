import os
import sys
sys.path.append(os.path.abspath("./sam2"))

from hydra import compose
from hydra.utils import instantiate

from omegaconf import OmegaConf

from sam2.build_sam import _load_checkpoint

def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.modeling.sam2_base.SAM2Base",
        "++model.image_encoder._target_=models.sam2.image_encoder.ImageEncoder",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    hydra_overrides.extend(hydra_overrides_extra)
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_image_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """Build SAM2 image predictor using SAM2Base (not video predictor)"""
    hydra_overrides = [
        "++model._target_=sam2.modeling.sam2_base.SAM2Base",
        "++model.image_encoder._target_=models.sam2.image_encoder.ImageEncoder",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    hydra_overrides.extend(hydra_overrides_extra)
    
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    
    if mode == "eval":
        model.eval()
        
        # Ensure deterministic behavior
        import torch
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Disable dropout
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    
    return model

def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,  # Deprecated parameter, kept for compatibility
    disable_compile=False,  # Option to disable torch.compile for deterministic behavior
    **kwargs,
):
    # Always use the custom SAM2VideoPredictor which now inherits from SAM2VideoPredictorVOS
    hydra_overrides = [
        "++model._target_=models.sam2.sam2_video_predictor.SAM2VideoPredictor",
        "++model.image_encoder._target_=models.sam2.image_encoder.ImageEncoder",
    ]
    
    # Add disable_compile parameter if requested
    if disable_compile:
        hydra_overrides.append("++model.disable_compile=true")
    
    # Note: vos_optimized parameter is no longer used since SAM2VideoPredictor 
    # now inherits from SAM2VideoPredictorVOS by default

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
        
        # Ensure deterministic behavior
        import torch
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Disable dropout
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    
    return model