// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import spherical_harmonics;

static const float eps = 1e-7;

[Differentiable]
float read_t1_float(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

[BackwardDerivative(bwd_read_t1_float_wavesum)]
[Differentiable]
float read_t1_float_wavesum(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

void bwd_read_t1_float_wavesum(uint32_t idx, DiffTensorView t1, float d_output)
{
    float total_d_output = WaveActiveSum(d_output);
    if (WaveIsFirstLane()) {
        bwd_diff(read_t1_float)(idx, t1, total_d_output);
    }
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[BackwardDerivative(bwd_read_t3_float3_wavesum)]
[Differentiable]
float3 read_t3_float3_wavesum(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

void bwd_read_t3_float3_wavesum(uint32_t idx, DiffTensorView t3, float3 d_output)
{   
    float3 total_d_output = WaveActiveSum(d_output);
    if (WaveIsFirstLane())
    {
        bwd_diff(read_t3_float3)(idx, t3, total_d_output);
    }
}

[Differentiable]
float4 read_t4_float4(uint32_t idx, DiffTensorView t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

[Differentiable]
float4 read_t2x2_float4(uint32_t idx, DiffTensorView t22)
{
    return float4(t22[uint3(idx, 0, 0)],
                  t22[uint3(idx, 1, 0)],
                  t22[uint3(idx, 0, 1)],
                  t22[uint3(idx, 1, 1)]);
}

[Differentiable]
float2x2 read_t2x2_float2x2(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}

[BackwardDerivative(bwd_read_t2x2_float2x2_wavesum)]
[Differentiable]
float2x2 read_t2x2_float2x2_wavesum(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}


void bwd_read_t2x2_float2x2_wavesum(uint32_t idx, DiffTensorView t2x2, float2x2 d_output)
{
    float2x2 total_d_output = WaveActiveSum(d_output);
    if (WaveIsFirstLane()) {
        bwd_diff(read_t2x2_float2x2)(idx, t2x2, total_d_output);
    }
}


[Differentiable]
float3x3 read_t3x3_float3x3(uint32_t idx, DiffTensorView t3x3)
{
    return float3x3(t3x3[uint3(idx, 0, 0)],
                    t3x3[uint3(idx, 1, 0)],
                    t3x3[uint3(idx, 2, 0)],
                    t3x3[uint3(idx, 0, 1)],
                    t3x3[uint3(idx, 1, 1)],
                    t3x3[uint3(idx, 2, 1)],
                    t3x3[uint3(idx, 0, 2)],
                    t3x3[uint3(idx, 1, 2)],
                    t3x3[uint3(idx, 2, 2)]);
}

[BackwardDerivative(bwd_read_t3x3_float3x3_wavesum)]
[Differentiable]
float3x3 read_t3x3_float3x3_wavesum(uint32_t idx, DiffTensorView t3x3)
{
    return float3x3(t3x3[uint3(idx, 0, 0)],
                    t3x3[uint3(idx, 1, 0)],
                    t3x3[uint3(idx, 2, 0)],
                    t3x3[uint3(idx, 0, 1)],
                    t3x3[uint3(idx, 1, 1)],
                    t3x3[uint3(idx, 2, 1)],
                    t3x3[uint3(idx, 0, 2)],
                    t3x3[uint3(idx, 1, 2)],
                    t3x3[uint3(idx, 2, 2)]);
}

void bwd_read_t3x3_float3x3_wavesum(uint32_t idx, DiffTensorView t3x3, float3x3 d_output)
{
    float3x3 total_d_output = WaveActiveSum(d_output);
    if (WaveIsFirstLane()) {
        bwd_diff(read_t3x3_float3x3)(idx, t3x3, total_d_output);
    }
}




[Differentiable]
float ndc2pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

[Differentiable]
float pix2ndc(float v, int S)
{
    return (2.0 * v + 1.0) / S - 1.0;
}

float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}

struct SigmaRS : IDifferentiable 
{
    float3x3 Sigma;
    float3x3 R;
    float3x3 S;
};

struct Camera : IDifferentiable
{
    float4x4 world_view_transform;
    float4x4 proj_mat;
    float3 position;
    float fovy;
    float fovx;
    float fx;
    float fy;
    int H;
    int W;
    
}

Camera load_camera(TensorView<float> world_view_transform_t, TensorView<float> proj_mat_t, TensorView<float> position_t, no_diff float fovy, no_diff float fovx, no_diff float fx, no_diff float fy, uint H, uint W) {
    float4x4 world_view_transform = float4x4(world_view_transform_t[uint2(0, 0)], world_view_transform_t[uint2(0, 1)], world_view_transform_t[uint2(0, 2)], world_view_transform_t[uint2(0, 3)],
                                             world_view_transform_t[uint2(1, 0)], world_view_transform_t[uint2(1, 1)], world_view_transform_t[uint2(1, 2)], world_view_transform_t[uint2(1, 3)],
                                             world_view_transform_t[uint2(2, 0)], world_view_transform_t[uint2(2, 1)], world_view_transform_t[uint2(2, 2)], world_view_transform_t[uint2(2, 3)],
                                             world_view_transform_t[uint2(3, 0)], world_view_transform_t[uint2(3, 1)], world_view_transform_t[uint2(3, 2)], world_view_transform_t[uint2(3, 3)]);

    float4x4 proj_mat = float4x4(proj_mat_t[uint2(0, 0)], proj_mat_t[uint2(0, 1)], proj_mat_t[uint2(0, 2)], proj_mat_t[uint2(0, 3)],
                                 proj_mat_t[uint2(1, 0)], proj_mat_t[uint2(1, 1)], proj_mat_t[uint2(1, 2)], proj_mat_t[uint2(1, 3)],
                                 proj_mat_t[uint2(2, 0)], proj_mat_t[uint2(2, 1)], proj_mat_t[uint2(2, 2)], proj_mat_t[uint2(2, 3)],
                                 proj_mat_t[uint2(3, 0)], proj_mat_t[uint2(3, 1)], proj_mat_t[uint2(3, 2)], proj_mat_t[uint2(3, 3)]);
    float3 position = float3(position_t[0], position_t[1], position_t[2]);

    return { world_view_transform, proj_mat, position, fovy, fovx, fx, fy, H, W};
}

[Differentiable]
float3 geom_transform_points(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz / (p_out.w + eps);
}

[Differentiable]
float3 geom_transform_points2(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz;
}

[Differentiable]
float3 project_point(float3 point, Camera cam) {
    float3 proj_point = geom_transform_points(point, mul(cam.proj_mat, cam.world_view_transform));
    float3 view_point = geom_transform_points2(point, cam.world_view_transform);
    proj_point.z = view_point.z;
    return proj_point;
}

[Differentiable]
float3x3 compute_jacobian(float3 xyz_ws, Camera cam) {
    float tan_half_fovx = tan(cam.fovx / 2.0);
    float tan_half_fovy = tan(cam.fovy / 2.0);
    float h_x = cam.W / (2.0 * tan_half_fovx);
    float h_y = cam.H / (2.0 * tan_half_fovy);

    float3 t = geom_transform_points(xyz_ws, cam.world_view_transform);

    // TODO: Clean this up, quite unintuive math and code
    const float limx = 1.3f * tan_half_fovx;
    const float limy = 1.3f * tan_half_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    float3x3 J = float3x3(h_x / t.z, 0.0, -(h_x * t.x) / (t.z * t.z),
                          0.0, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
                          0.0, 0.0, 0.0);

    return J;
}

[Differentiable]
float2x2 covariance_3d_to_2d(Camera cam, float3 xyz_ws, float3x3 cov_ws) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 J = compute_jacobian(xyz_ws, cam);
    float3x3 cov_vs = mul(J, mul(R, mul(cov_ws, mul(transpose(R), transpose(J)))));
    cov_vs[0][0] += 0.3;
    cov_vs[1][1] += 0.3;

    return float2x2(cov_vs);
}

[Differentiable]
float3x3 covariance_3dw_to_3dv(Camera cam, float3x3 cov_ws) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 cov_vs = mul(R, mul(cov_ws, transpose(R)));
    
    return float3x3(cov_vs);
}

struct Gaussian_3D : IDifferentiable
{
    float3 xyz_ws;
    SpherHarmCoeffs sh_coeffs;
    float4 rotations;
    float3 scales;
};

[Differentiable]
Gaussian_3D load_gaussian(int32_t g_idx,
                          DiffTensorView xyz_ws,
                          DiffTensorView sh_coeffs,
                          DiffTensorView rotations,
                          DiffTensorView scales,
                          uint active_sh)
{
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);
    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh);
    float4 g_rotations = read_t4_float4(g_idx, rotations);
    float3 g_scales = read_t3_float3(g_idx, scales);

    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales };
}


struct Splat_2D_Vertex : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float2x2 cov_vs;
};

float splat_radius(float2x2 cov_vs, float det) {
    float mid = 0.5f * (cov_vs[0][0] + cov_vs[1][1]);
    float eigen_val_1 = mid + sqrt(max(0.1f, mid * mid - det));
    float eigen_val_2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(eigen_val_1, eigen_val_2)));

    return radius;
}

float splat_radius2(float3 xyz_vs, float density, float thresh, float3x3 Sw, float3x3 invCov3dv, Camera cam) {
    float3 mu = xyz_vs;
    // float3 dir_mu = normalize(xyz_vs);
    float3x3 S = Sw; // mul(W, mul(Sw, transpose(W)));

    // TODO: This s_app limit logic should incorporate the directionality, it should'nt be same for all orientations of the Gaussian.
    // float s_app = 1.0f/dot(dir_mu, mul(invCov3dv, dir_mu));
    // float s_app = max(S[0][0], max(S[1][1], S[2][2])); // Geometric mean of diagonal elements
    float s_app = pow(S[0][0] * S[1][1] * S[2][2], 1.0/3.0); // Geometric mean of diagonal elements
    float pdf_lim = log(1.0/(1-thresh)) * 1.0/(density * s_app * 2.506f);
    // NOTE: Have removed pdf_lim > 1 condition.
    // printf("2 * log(pdf_lim): %f, muSmu: %f\n", 2 * log(pdf_lim), dot(mu, mul(invCov3dv, mu)));
    float c =  dot(mu, mul(invCov3dv, mu)) + 2 * log(pdf_lim);

    float3 v = mul(invCov3dv, mu);
    float3x3 vvT = float3x3(
        v.x * v.x, v.x * v.y, v.x * v.z,
        v.y * v.x, v.y * v.y, v.y * v.z,
        v.z * v.x, v.z * v.y, v.z * v.z
);
    
    // [A, B/2, D/2; B/2, C, E/2; D/2, E/2, F]
    float3x3 Aq = invCov3dv - vvT/c;
    float A = Aq[0][0];
    float B = 2*Aq[0][1];
    float D = 2*Aq[0][2];
    float C = Aq[1][1];
    float E = 2*Aq[1][2];
    float F = Aq[2][2];

    float fx = cam.fx;
    float fy = cam.fy;
    float cx = cam.W / 2.0;
    float cy = cam.H / 2.0;

    float A_ = A / (fx*fx);
    float B_ = B / (fx*fy);
    float C_ = C / (fy*fy);
    float D_ = (D / fx) - (2*A_*cx) - (B_*cy);
    float E_ = (E / fy) - (2*C_*cy) - (B_*cx);
    float F_ = F  + A_*cx*cx + C_*cy*cy + B_*cx*cy - (D * cx/fx) - (E * cy/fy);
    
    // TODO: Combine this into a single operation -- don't recreate matrix and then compute det. Compute it directly.
    // float3x3 Aq_ = float3x3(A_, B_ / 2, D_ / 2, B_ / 2, C_, E_ / 2, D_ / 2, E_ / 2, F_);
    float det_Aq_ = (A_ * C_ - (B_ * B_) / 4.0) * F_ + (B_ * D_ * E_ - C_*D_*D_ - A_*E_*E_)/4.0;
    // float det_Aq_old = compute_det3x3(Aq_);
    float K = -(det_Aq_ / (A_ * C_ - (B_ * B_)/4.0));

    float A_1 = A_ / K;
    float B_1 = B_ / K;
    float C_1 = C_ / K;

    float lmin = ((A_1 + C_1) - sqrt((A_1 - C_1) * (A_1 - C_1) + (B_1 * B_1))) / 2.0;
    if (lmin < 0.0) {
        return 0.0;
    }
    float radius = ceil(1.0 / sqrt(lmin));
    return radius;
}

[Differentiable]
float compute_det(float2x2 M) {
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}

[Differentiable]
float compute_det3x3(float3x3 M) {
    return M[0][0] * (M[1][1] * M[2][2] - M[2][1] * M[1][2]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

[Differentiable]
float3x3 inverse3x3(float3x3 M)
{
    float det = M[0][0] * (M[1][1] * M[2][2] - M[2][1] * M[1][2]) -
                M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    // Add a small epsilon to avoid division by zero
    float epsilon = 1e-10;
    float invDet = 1.0 / det;
    // float invDet = 1.0 / (det + epsilon);

    // // Check if the determinant is close to zero
    // if (abs(det) < epsilon) {
    //     // Return identity matrix if the input is singular
    //     return float3x3(1.0, 0.0, 0.0,
    //                     0.0, 1.0, 0.0,
    //                     0.0, 0.0, 1.0);
    // }

    float3x3 minv;
    minv[0][0] = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * invDet;
    minv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invDet;
    minv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet;
    minv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invDet;
    minv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet;
    minv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invDet;
    minv[2][0] = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * invDet;
    minv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invDet;
    minv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invDet;

    return minv;
}



[Differentiable]
Splat_2D_Vertex load_splat_vertex(int32_t g_idx,
                                  DiffTensorView xyz_vs,
                                  DiffTensorView cov_vs,
                                  DiffTensorView rgb)
{
    float3 g_xyz_vs = read_t3_float3(g_idx, xyz_vs);
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float2x2 g_cov_vs = read_t2x2_float2x2(g_idx, cov_vs);

    return { g_xyz_vs, g_rgb, g_cov_vs };
}

[Differentiable]
Splat_2D_Vertex project_gaussian_to_camera(Gaussian_3D g, Camera cam, uint active_sh, bool softplus_rgb) {
    float3 xyz_vs = project_point(g.xyz_ws, cam);
    if (xyz_vs.z <= 0.2) {
        return {float3(0.0), float3(0.0), float2x2(0.0)};
    }
    float3 rgb = softplus_rgb ? compute_color_from_sh_coeffs_softplus(g.sh_coeffs, g.xyz_ws, cam.position, active_sh)
                              : compute_color_from_sh_coeffs(g.sh_coeffs, g.xyz_ws, cam.position, active_sh);
    SigmaRS sigma_rs = get_covariance_from_quat_scales(g.rotations, g.scales);
    float3x3 cov_ws = sigma_rs.Sigma;
    float2x2 cov_vs = covariance_3d_to_2d(cam, g.xyz_ws, cov_ws);

    return { xyz_vs, rgb, cov_vs };
}

// Function for new tiling shader, doesn't need to compute 2D covariance. 
[Differentiable]
Splat_2D_Vertex project_gaussian_to_camera_newshader(Gaussian_3D g, Camera cam, uint active_sh, bool softplus_rgb) {
    float3 xyz_vs = project_point(g.xyz_ws, cam);
    if (xyz_vs.z <= 0.2) {
        return { float3(0.0), float3(0.0), float2x2(0.0) };
    }
    float3 rgb = softplus_rgb ? compute_color_from_sh_coeffs_softplus(g.sh_coeffs, g.xyz_ws, cam.position, active_sh)
                              : compute_color_from_sh_coeffs(g.sh_coeffs, g.xyz_ws, cam.position, active_sh);
    SigmaRS sigma_rs = get_covariance_from_quat_scales(g.rotations, g.scales);
    float3x3 cov_ws = sigma_rs.Sigma;
    float2x2 cov_vs = {0.0, 0.0, 0.0, 0.0}; //Because it's not used in the new tile computation code

    return { xyz_vs, rgb, cov_vs };
}


struct Splat_2D_AlphaBlend : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float opacity;
    float2x2 inv_cov_vs;
};

// [BackwardDerivative(bwd_load_splat_alphablend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
[Differentiable]
Splat_2D_AlphaBlend load_splat_alphablend(int32_t g_idx,
                                          DiffTensorView xyz_vs,
                                          DiffTensorView inv_cov_vs,
                                          DiffTensorView opacity,
                                          DiffTensorView rgb)
{
    // float3 g_xyz_vs = read_t3_float3_wavesum(g_idx, xyz_vs);
    // float3 g_rgb = read_t3_float3_wavesum(g_idx, rgb);
    // float g_opacity = read_t1_float_wavesum(g_idx, opacity);
    // float2x2 g_inv_cov = read_t2x2_float2x2_wavesum(g_idx, inv_cov_vs);

    float3 g_xyz_vs = read_t3_float3(g_idx, xyz_vs);
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float g_opacity = read_t1_float(g_idx, opacity);
    float2x2 g_inv_cov = read_t2x2_float2x2(g_idx, inv_cov_vs);

    return { g_xyz_vs, g_rgb, g_opacity, g_inv_cov };
}




[Differentiable]
float4 evaluate_splat(Splat_2D_AlphaBlend g, 
                      float2 pix_coord,
                      uint32_t H,
                      uint32_t W)
{

    float3 g_xyz = g.xyz_vs;
    float3 g_rgb = g.rgb;
    float g_opacity = g.opacity;
    float2x2 g_inv_cov_vs = g.inv_cov_vs;

    float2 d = { pix_coord.x - ndc2pix(g_xyz.x, W),
                 pix_coord.y - ndc2pix(g_xyz.y, H) };
    float power = -0.5f * (g_inv_cov_vs[0][0] * d.x * d.x +
                           g_inv_cov_vs[1][1] * d.y * d.y + (g_inv_cov_vs[0][1] + g_inv_cov_vs[1][0]) * d.x * d.y);
    float alpha = min(0.99f, g_opacity * exp(power));
    float3 premult_rgb = g_rgb * alpha;
    
    return float4(premult_rgb, alpha);
}

struct Splat_3D_AlphaBlend : IDifferentiable
{
    float3 xyz3d_vs;
    float3 rgb;
    float opacity; // Ranges between 0 and 1, same as 3DGS Opacity
    float opacity_volr; // Ranges between 0 and max_opacity_volr, controls density of the Gaussian
    float3x3 inv_cov3d_vs;
};

[Differentiable]
Splat_3D_AlphaBlend load_splat3d_alphablend(int32_t g_idx,
                                          DiffTensorView xyz3d_vs,
                                          DiffTensorView inv_cov3d_vs,
                                          DiffTensorView opacity,
                                          DiffTensorView opacity_volr,
                                          DiffTensorView rgb)
{
    float3 g_xyz3d_vs = read_t3_float3(g_idx, xyz3d_vs);
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float g_opacity = read_t1_float(g_idx, opacity);
    float g_opacity_volr = read_t1_float(g_idx, opacity_volr);
    float3x3 g_inv_cov3d = read_t3x3_float3x3(g_idx, inv_cov3d_vs);

    // float3 g_xyz3d_vs = read_t3_float3_wavesum(g_idx, xyz3d_vs);
    // float3 g_rgb = read_t3_float3_wavesum(g_idx, rgb);
    // float g_opacity = read_t1_float_wavesum(g_idx, opacity);
    // float g_opacity_volr = read_t1_float_wavesum(g_idx, opacity_volr);
    // float3x3 g_inv_cov3d = read_t3x3_float3x3_wavesum(g_idx, inv_cov3d_vs);

    return { g_xyz3d_vs, g_rgb, g_opacity, g_opacity_volr, g_inv_cov3d };
}

[Differentiable]
float3 load_xyz3d_abs(int32_t g_idx,
                      DiffTensorView xyz) {        
    return read_t3_float3(g_idx, xyz);
}

[Differentiable]
float compute_maxpdf_ray(float3 mu,
                         float3x3 inv_cov3d,
                         float3 d) 
{
    float denominator = max(eps*100.0f, dot(d, mul(inv_cov3d, d)));
    float numerator = dot(mu, mul(inv_cov3d, d));
    float tmax = max(numerator / denominator, 0.0f); 
                 
    float3 diff = tmax * d - mu; 
    float power = -0.5 * dot(diff, mul(inv_cov3d, diff));
    float pdfmax = exp(power);
    return pdfmax;
}


[Differentiable]
float4 evaluate_splat_volr(Splat_3D_AlphaBlend g, 
                      float3 d,
                      uint32_t H,
                      uint32_t W, 
                      no_diff float fx,
                      no_diff float fy)
{

    float3 g_xyz = g.xyz3d_vs;
    float3 g_rgb = g.rgb;
    float g_opacity = g.opacity;
    float g_opacity_volr = g.opacity_volr;
    float3x3 g_inv_cov3d = g.inv_cov3d_vs;

    float denominator = max(eps, dot(d, mul(g_inv_cov3d, d)));
    float numerator = dot(g_xyz, mul(g_inv_cov3d, d));
    float tmax = numerator / denominator;
    if (numerator <= 0.0f){
       return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    // Added this condition later, added to avoid NaN due to very large / very small values in pdfmax power
    // if (tmax == 0.0f) {
    //     return float4(0.0f, 0.0f, 0.0f, 0.0f);
    // }

    float3 diff = tmax * d - g_xyz;
    float power = -0.5 * dot(diff, mul(g_inv_cov3d, diff));
    float pdfmax = exp(power);

    float sigma_d = sqrt(1.0f / (denominator + eps));
    // sqrt(2 * 3.14159) = 2.506
    float alpha = 1.0f - exp(-2.506f * g_opacity_volr * sigma_d * pdfmax);

    // float3 mu_norm = normalize(g_xyz);
    // float denominator_scale = max(eps, dot(mu_norm, mul(g_inv_cov3d, mu_norm)));
    // float sigma_d_scale = sqrt(1.0f / denominator_scale);
    // float norm_scale = 1.0f - exp(-sqrt(2*3.14159f) * g_opacity_volr * sigma_d_scale);
    // alpha = alpha / norm_scale;

    alpha = min(0.99f, alpha);
    float3 premult_rgb = g_rgb * alpha;
    
    return float4(premult_rgb, alpha);
}

[Differentiable]
float evaluate_splat_tomo(Splat_3D_AlphaBlend g,
                           float2 pix_coord,
                           uint32_t H,
                           uint32_t W,
                           no_diff float fovx,
                           no_diff float fovy)
{

    float3 g_xyz = g.xyz3d_vs;
    float3 g_rgb = g.rgb;
    float g_opacity = g.opacity;
    float g_opacity_volr = g.opacity_volr;
    float3x3 g_inv_cov3d = g.inv_cov3d_vs;
    // TODO: Till d = normalize(d), bring this computation outside of this function, probably causing slowdown.
    float tan_half_fovx = tan(fovx / 2.0f);
    float tan_half_fovy = tan(fovy / 2.0f);
    float fx = W / (2.0f * tan_half_fovx);
    float fy = H / (2.0f * tan_half_fovy);
    float3 d = { (pix_coord.x - W / 2.0f) / fx, (pix_coord.y - H / 2.0f) / fy, 1.0f };
    d = normalize(d);

    float denominator = max(eps, dot(d, mul(g_inv_cov3d, d)));
    float numerator = dot(g_xyz, mul(g_inv_cov3d, d));
    float tmax = max(numerator / denominator, 0.0f);

    float3 diff = tmax * d - g_xyz;
    float power = -0.5 * dot(diff, mul(g_inv_cov3d, diff));
    float pdfmax = exp(power);

    float sigma_d = sqrt(1.0f / denominator);
    // float alpha = g_opacity * (1.0f - exp(-sqrt(2 * 3.14159f) * g_opacity_volr * sigma_d * pdfmax));
    float alpha = sqrt(2 * 3.14159f) * g_opacity_volr * sigma_d * pdfmax;
    
    return alpha;
}


[Differentiable]
SigmaRS get_covariance_from_quat_scales(float4 q, float3 s) {
    float r = q[0], x = q[1], y = q[2], z = q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

    float3x3 scales_matrix = float3x3(s[0], 0, 0,
                                      0, s[1], 0,
                                      0, 0, s[2]);

    float3x3 L = mul(rotation_matrix, scales_matrix);

    return {mul(L, transpose(L)), rotation_matrix, scales_matrix};
}

// [Differentiable]
// float3x3 get_covariance_from_quat_scales(float4 q, float3 s) {
//     float r = q[0], x = q[1], y = q[2], z = q[3];

//     float3x3 rotation_matrix = float3x3(
//         1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
//         2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
//         2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

//     float3x3 scales_matrix = float3x3(s[0], 0, 0,
//                                       0, s[1], 0,
//                                       0, 0, s[2]);

//     float3x3 L = mul(rotation_matrix, scales_matrix);

//     return mul(L, transpose(L));
// }


[Differentiable]
float3x3 get_inverse_covariance_from_quat_scales(float4 q, float3 s) {
    float r = q[0], x = q[1], y = q[2], z = q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

    // float3x3 scales_matrix = float3x3(min(1000.0f, 1.0f / s[0]), 0, 0,
    //                                   0, min(1000.0f, 1.0f / s[1]), 0,
    //                                   0, 0, min(1000.0f, 1.0f / s[2]));
    
    float3x3 scales_matrix = float3x3(1.0f / s[0], 0, 0,
                                      0, 1.0f / s[1], 0,
                                      0, 0, 1.0f / s[2]);
    float3x3 L = mul(rotation_matrix, scales_matrix);

    return mul(L, transpose(L));
}