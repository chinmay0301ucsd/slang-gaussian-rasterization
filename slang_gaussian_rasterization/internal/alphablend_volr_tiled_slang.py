# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from slang_gaussian_rasterization.internal.render_grid import RenderGrid
import slang_gaussian_rasterization.internal.slang.slang_modules as slang_modules
from slang_gaussian_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def render_alpha_blend_volr_tiles_slang_raw(xyz_ws, rotations, scales, opacity,
                                       opacity_volr, sh_coeffs, active_sh,
                                       world_view_transform, proj_mat, cam_pos,
                                       fovy, fovx, height, width, tile_size=16):
    
    render_grid = RenderGrid(height,
                             width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    
    scale3d_factor = torch.max(torch.sqrt(2 * torch.log(opacity_volr / 0.011) / 9.0), torch.ones_like(opacity_volr))
    sorted_gauss_idx, tile_ranges, radii, xyz_vs, xyz3d_cam, inv_cov3d_vs, inv_cov3d_vs, rgb  = vertex_and_tile_shader(xyz_ws,
                                                                                           rotations,
                                                                                           scales,
                                                                                           scale3d_factor,
                                                                                           sh_coeffs,
                                                                                           active_sh,
                                                                                           world_view_transform,
                                                                                           proj_mat,
                                                                                           cam_pos,
                                                                                           fovy,
                                                                                           fovx,
                                                                                           render_grid)
   
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        xyz_vs.retain_grad()
    except:
        pass
    
    image_rgb = AlphaBlendVolrTiledRender.apply(
        sorted_gauss_idx,
        tile_ranges,
        xyz3d_cam,
        inv_cov3d_vs,
        opacity,
        opacity_volr,
        rgb,
        render_grid,
        fovx,
        fovy
        )
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'viewspace_points': xyz_vs,
        'visibility_filter': radii > 0,
        'radii': radii,
    }

    return render_pkg


class AlphaBlendVolrTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sorted_gauss_idx, tile_ranges, xyz3d_vs, inv_cov3d_vs,
                 opacity, opacity_volr, rgb, render_grid, fovx, fovy, device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4), 
                                 device=device)
        n_contributors = torch.zeros((render_grid.image_height, 
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_volr_shaders[(render_grid.tile_height, render_grid.tile_width)]
        splat_kernel_with_args = alpha_blend_tile_shader.splat_volr_tiled(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz3d_vs=xyz3d_vs, 
            inv_cov3d_vs=inv_cov3d_vs, 
            opacity=opacity,
            opacity_volr=opacity_volr, 
            rgb=rgb, 
            output_img=output_img,
            n_contributors=n_contributors,
            fovx=fovx,
            fovy=fovy,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        ctx.save_for_backward(sorted_gauss_idx, tile_ranges,
                              xyz3d_vs, inv_cov3d_vs, opacity, opacity_volr, rgb, 
                              output_img, n_contributors)
        ctx.render_grid = render_grid
        ctx.fovx = fovx
        ctx.fovy = fovy
        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (sorted_gauss_idx, tile_ranges, 
         xyz3d_vs, inv_cov3d_vs, opacity, opacity_volr, rgb, 
         output_img, n_contributors) = ctx.saved_tensors
        render_grid = ctx.render_grid
        fovx, fovy = ctx.fovx, ctx.fovy
        xyz3d_vs_grad = torch.zeros_like(xyz3d_vs)
        inv_cov3d_vs_grad = torch.zeros_like(inv_cov3d_vs)
        opacity_grad = torch.zeros_like(opacity)
        opacity_volr_grad = torch.zeros_like(opacity_volr)
        rgb_grad = torch.zeros_like(rgb)


        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_volr_shaders[(render_grid.tile_height, render_grid.tile_width)]

        kernel_with_args = alpha_blend_tile_shader.splat_volr_tiled.bwd(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz3d_vs=(xyz3d_vs, xyz3d_vs_grad),
            inv_cov3d_vs=(inv_cov3d_vs, inv_cov3d_vs_grad),
            opacity=(opacity, opacity_grad),
            opacity_volr=(opacity_volr, opacity_volr_grad),
            rgb=(rgb, rgb_grad),
            output_img=(output_img, grad_output_img),
            n_contributors=n_contributors,
            fovx=fovx,
            fovy=fovy,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width,
            )
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        
        return None, None, xyz3d_vs_grad, inv_cov3d_vs_grad, opacity_grad, opacity_volr_grad, rgb_grad, None, None, None
