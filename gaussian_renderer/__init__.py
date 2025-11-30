#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat

import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel

def build_viewmat_from_camera(viewpoint_camera):
    """
    构建gsplat所需的viewmat矩阵 (world-to-camera)
    Args:
        viewpoint_camera: Camera对象，包含world_view_transform
    Returns:
        viewmats: [1, 4, 4] tensor, world-to-camera变换矩阵
    """
    # world_view_transform已经是world-to-camera，但需要转置回来
    # 因为Octree-GS中存储的是转置后的版本
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    return viewmat.unsqueeze(0)  # [1, 4, 4]

def build_K_matrix(viewpoint_camera):
    """
    构建gsplat所需的相机内参矩阵K
    Args:
        viewpoint_camera: Camera对象，包含FoVx, FoVy, image_width, image_height
    Returns:
        Ks: [1, 3, 3] tensor, 相机内参矩阵
    """
    # 从FoV计算焦距
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    focal_x = viewpoint_camera.image_width / (2.0 * tanfovx)
    focal_y = viewpoint_camera.image_height / (2.0 * tanfovy)
    
    # 主点通常在图像中心
    cx = viewpoint_camera.image_width / 2.0
    cy = viewpoint_camera.image_height / 2.0
    
    # 构建K矩阵
    K = torch.tensor([
        [focal_x, 0.0, cx],
        [0.0, focal_y, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device="cuda")
    
    return K.unsqueeze(0)  # [1, 3, 3]


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)
            
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets 

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False, ape_code=-1):
    """
    Render the scene using gsplat backend.
    
    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # 构建gsplat所需的相机参数
    viewmats = build_viewmat_from_camera(viewpoint_camera)
    Ks = build_K_matrix(viewpoint_camera)
    
    # 准备gsplat所需的参数
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    
    # 调整scales以应用scaling_modifier
    scales = scaling * scaling_modifier
    
    # 转换四元数为gsplat格式 (wxyz)
    # Octree-GS使用的rotation_activation已经输出wxyz格式的四元数
    quats = rot
    
    # 调用gsplat.rasterization
    # 使用packed=True模式以优化内存使用
    render_colors, render_alphas, meta = rasterization(
        means=xyz,                    # [N, 3]
        quats=quats,                  # [N, 4] wxyz
        scales=scales,                # [N, 3]
        opacities=opacity.squeeze(-1), # [N]
        colors=color,                 # [N, 3]
        viewmats=viewmats,            # [1, 4, 4]
        Ks=Ks,                        # [1, 3, 3]
        width=width,
        height=height,
        packed=True,                  # 使用packed模式优化内存
        backgrounds=bg_color.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1, 1, 1, 3]
    )
    
    # 提取渲染结果
    # render_colors shape: [1, height, width, 3]
    # 转换为 [3, height, width] 格式以匹配原始输出
    rendered_image = render_colors[0].permute(2, 0, 1)  # [3, H, W]
    
    # 从meta中提取radii信息
    # gsplat返回的radii shape: [C, N] 或稀疏格式
    radii = meta.get("radii", torch.zeros(xyz.shape[0], dtype=torch.int32, device="cuda"))
    if radii.dim() > 1:
        radii = radii[0]  # 取第一个相机的radii
    
    # 创建visibility filter
    # gsplat在packed模式下会返回gaussian_ids，指示哪些高斯是可见的
    if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
        # packed模式：需要从sparse到dense
        gaussian_ids = meta["gaussian_ids"]
        if gaussian_ids.dim() > 1:
            gaussian_ids = gaussian_ids[0]  # [M] M是可见高斯数量
        
        # 创建visibility filter
        visibility_filter = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
        visibility_filter[gaussian_ids] = True
        
        # 更新radii为完整版本
        radii_full = torch.zeros(xyz.shape[0], dtype=radii.dtype, device="cuda")
        if radii.numel() > 0:
            radii_full[gaussian_ids] = radii[:len(gaussian_ids)]
        radii = radii_full
    else:
        # 非packed模式或fallback
        visibility_filter = radii > 0
    
    # 使用meta中的means2d作为screenspace_points（如果需要梯度）
    if "means2d" in meta and meta["means2d"] is not None:
        means2d = meta["means2d"]
        if means2d.dim() > 2:
            means2d = means2d[0]  # [N, 2]
        # 将2D坐标扩展到3D以匹配原始screenspace_points格式
        # 注意：gsplat的means2d已经包含了梯度计算
        if screenspace_points.shape[0] == means2d.shape[0]:
            screenspace_points[:, :2] = means2d

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : visibility_filter,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : visibility_filter,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Prefilter voxels using gsplat rasterization for visibility culling.
    
    Background tensor (bg_color) must be on GPU!
    """

    # 获取被anchor_mask选中的anchor点
    means3D = pc.get_anchor[pc._anchor_mask]
    
    # 获取scales和rotations
    if pipe.compute_cov3D_python:
        # 如果需要预计算协方差矩阵，使用cov3D
        # 注意：gsplat也支持直接传入cov3D矩阵
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]
        # 暂时不使用预计算的协方差，因为gsplat更喜欢scales+quats
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]
    
    # 调整scales
    scales = scales[:, :3] * scaling_modifier
    
    # 构建gsplat所需的相机参数
    viewmats = build_viewmat_from_camera(viewpoint_camera)
    Ks = build_K_matrix(viewpoint_camera)
    
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    
    # 创建虚拟的opacities和colors（仅用于可见性检测）
    opacities = torch.ones(means3D.shape[0], device="cuda")
    colors = torch.ones(means3D.shape[0], 3, device="cuda")
    
    # 调用gsplat.rasterization进行可见性过滤
    # 使用packed=True可以直接获取可见高斯的IDs
    try:
        _, _, meta = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=True,
            backgrounds=bg_color.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            render_mode="RGB",  # 只需要基本渲染模式
        )
        
        # 从meta中提取可见的gaussian_ids
        if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
            gaussian_ids = meta["gaussian_ids"]
            if gaussian_ids.dim() > 1:
                gaussian_ids = gaussian_ids[0]
            
            # 创建visible_mask
            # 首先克隆原始的anchor_mask
            visible_mask = pc._anchor_mask.clone()
            
            # 创建临时mask标记哪些被_anchor_mask选中的anchor是可见的
            temp_visible = torch.zeros(means3D.shape[0], dtype=torch.bool, device="cuda")
            temp_visible[gaussian_ids] = True
            
            # 更新visible_mask
            visible_mask[pc._anchor_mask] = temp_visible
        else:
            # 如果没有gaussian_ids，使用radii作为fallback
            radii = meta.get("radii", None)
            if radii is not None:
                if radii.dim() > 1:
                    radii = radii[0]
                visible_mask = pc._anchor_mask.clone()
                visible_mask[pc._anchor_mask] = radii > 0
            else:
                # 最坏情况：保持原始mask
                visible_mask = pc._anchor_mask.clone()
                
    except Exception as e:
        # 如果gsplat调用失败，fallback到保守策略
        print(f"Warning: gsplat prefilter failed with error: {e}, using conservative mask")
        visible_mask = pc._anchor_mask.clone()
    
    return visible_mask

