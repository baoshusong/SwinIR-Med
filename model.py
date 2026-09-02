import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== SwinIR‑Med 网络模块 =====================
class PatchEmbed(nn.Module):
    """将CT图像Patch化并嵌入（单通道原生支持）"""
    def __init__(self, img_size=128, patch_size=1, in_chans=1, embed_dim=60):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def image_to_tokens(x):
    """将图像特征 (B,C,H,W) 展平为 token 序列 (B,N,C)"""
    return x.flatten(2).transpose(1, 2)

def tokens_to_image(x, H, W):
    """将 token 序列 (B,N,C) 还原为图像特征 (B,C,H,W)"""
    B, N, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)

class Mlp(nn.Module):
    """Swin Transformer中的MLP模块"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """窗口注意力机制（W‑MSA/SW‑MSA）"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer基础块"""
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0‑window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class HighFreqBranch(nn.Module):
    """高频分支：Swin Transformer块，保留边缘细节"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=2., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = nn.Sequential(*[
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=True,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    def forward(self, x, x_size):
        H, W = x_size
        res = x
        x = self.residual_group(x)
        x = tokens_to_image(x, H, W)
        x = self.conv(x)
        x = image_to_tokens(x)
        return x + res

class LowFreqBranch(nn.Module):
    """低频分支：大核深度卷积，保证平滑区域连续性，抑制伪影"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
    def forward(self, x, x_size):
        H, W = x_size
        res = x
        x = tokens_to_image(x, H, W)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = image_to_tokens(x)
        return x + res

class ResidualGroup(nn.Module):
    """高低频分治残差组（退化自适应 FiLM + 空间自适应门控融合）"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.high_freq_branch = HighFreqBranch(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer
        )
        self.low_freq_branch = LowFreqBranch(dim=dim)
        # 空间自适应门控：由高低频特征逐位置生成融合权重 gate∈[0,1]
        self.fusion_gate = nn.Conv2d(dim * 2, 1, kernel_size=1)
        self.fusion_conv = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, x, x_size, deg_scale=None, deg_shift=None):
        H, W = x_size
        res = x
        if deg_scale is not None:
            # 退化条件 FiLM：逐通道调制（deg_scale/shift 形状 (B,1,C)）
            x = x * (1 + deg_scale) + deg_shift
        high_feat = self.high_freq_branch(x, x_size)
        low_feat = self.low_freq_branch(x, x_size)
        high_2d = tokens_to_image(high_feat, H, W)
        low_2d = tokens_to_image(low_feat, H, W)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([high_2d, low_2d], dim=1)))  # (B,1,H,W)
        fused_feat = gate * high_2d + (1 - gate) * low_2d
        fused_feat = self.fusion_conv(fused_feat)
        fused_feat = fused_feat.flatten(2).transpose(1, 2)
        return fused_feat + res

class DegradationAwareModule(nn.Module):
    """盲退化建模模块（CT 显式退化空间）。

    将 LR 切片编码为低维退化表征 z，并映射为逐通道 FiLM 调制参数。
    退化空间显式覆盖 CT 采集的三要素：下采样核(kernel)、噪声水平(noise)、层厚(slice thickness)，
    由网络从数据中学习得到，用于驱动各残差组的退化自适应调制（blind SR）。
    """
    def __init__(self, in_chans=1, embed_dim=60, deg_dim=16):
        super().__init__()
        self.deg_dim = deg_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        # 退化表征投影：encoder 特征 -> 低维退化码 z
        self.code_fc = nn.Linear(128, deg_dim)
        # z -> FiLM(scale, shift)
        self.modulation_fc = nn.Sequential(
            nn.Linear(deg_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim * 2)
        )
    def forward(self, x):
        deg_feat = self.encoder(x).flatten(1)              # (B, 128)
        z = self.code_fc(deg_feat)                         # (B, deg_dim) 退化表征
        modulation_params = self.modulation_fc(z)
        scale, shift = modulation_params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return z, scale, shift

class SkipConnectionBlock(nn.Module):
    """U‑Net式跳跃连接，浅层特征上采样"""
    def __init__(self, in_channels, out_channels, scale_factor=4):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.upsample(x)
        x = self.smooth(x)
        return x

class SwinIRMed(nn.Module):
    """SwinIR‑Med：医学影像4x超分"""
    def __init__(self, img_size=128, patch_size=1, in_chans=3, out_chans=1, deg_dim=16,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.degradation_module = DegradationAwareModule(in_chans=1, embed_dim=embed_dim, deg_dim=deg_dim)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.img_size
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer)
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        if upscale == 4:
            # 两级 PixelShuffle 上采样：128 -> 256 -> 512，参数更少、更稳
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU()
            )
        elif upscale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale)
            )
        self.skip_connection = SkipConnectionBlock(embed_dim, embed_dim, self.upscale)
        self.fusion_skip = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, kernel_size=3, padding=1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward_features(self, x, deg_scale=None, deg_shift=None):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size, deg_scale, deg_shift)
        x = self.norm(x)
        x = tokens_to_image(x, x_size[0], x_size[1])
        return x
    def forward(self, x, return_deg=False):
        # 2.5D：仅用中心切片做全局残差（bicubic 上采样），保持输出为单通道
        c = self.in_chans // 2
        center = x[:, c:c + 1]
        global_residual = F.interpolate(center, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        x_first = self.conv_first(x)
        skip_feat = x_first
        # 退化编码：注入到每个残差组（token 空间形状 (B,1,C)）
        z, deg_scale, deg_shift = self.degradation_module(center)  # 对中心切片编码退化，z:(B,deg_dim)
        B, C = deg_scale.shape[0], deg_scale.shape[1]
        deg_scale_t = deg_scale.view(B, 1, C)
        deg_shift_t = deg_shift.view(B, 1, C)
        x_body = self.forward_features(x_first, deg_scale_t, deg_shift_t)
        x_body = self.conv_after_body(x_body)
        x = x_first + x_body
        x_upscale = self.upsample(x)
        skip_upscaled = self.skip_connection(skip_feat)
        fused = self.fusion_skip(torch.cat([x_upscale, skip_upscaled], dim=1))
        hr_residual = self.conv_last(fused)
        hr_img = global_residual + hr_residual
        if return_deg:
            return hr_img, z
        return hr_img

# ===================== 损失函数 =====================
class SwinIRMedLoss(nn.Module):
    """SwinIR‑Med 复合损失。

    在原有 像素L1 + 边缘(Sobel) + TV + 平滑 基础上，新增三类物理/临床驱动项：
      - HU 保真损失：在 HU 绝对空间回归（而非仅归一化窗宽空间），保证 CT 值可解释；
      - 下采样一致性损失：HR 预测按同一倍数下采样后应重建回观测 LR（自监督物理约束）；
      - 退化码一致性损失：HR 预测下采样后的退化表征应与观测 LR 的退化表征一致
        （与 DegradationAwareModule 的盲退化建模闭环，强化 #2）。
    """
    def __init__(self, use_edge_loss=True, edge_weight=0.01,
                 use_tv_loss=True, tv_weight=0.08,
                 use_smooth_loss=True, smooth_weight=0.04,
                 use_hu_loss=True, hu_weight=0.10,
                 use_consist_loss=True, consist_weight=0.05,
                 use_deg_loss=True, deg_weight=0.02,
                 upscale=4):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.use_edge_loss = use_edge_loss
        self.edge_weight = edge_weight
        self.use_tv_loss = use_tv_loss
        self.tv_weight = tv_weight
        self.use_smooth_loss = use_smooth_loss
        self.smooth_weight = smooth_weight
        self.use_hu_loss = use_hu_loss
        self.hu_weight = hu_weight
        self.use_consist_loss = use_consist_loss
        self.consist_weight = consist_weight
        self.use_deg_loss = use_deg_loss
        self.deg_weight = deg_weight
        self.upscale = upscale
        # Sobel 核注册为 buffer，避免每次前向重建
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    def gradient_loss(self, pred, gt):
        def sobel(x):
            kx = self.sobel_x.to(device=x.device, dtype=x.dtype)
            ky = self.sobel_y.to(device=x.device, dtype=x.dtype)
            edge_x = F.conv2d(x, kx, padding=1)
            edge_y = F.conv2d(x, ky, padding=1)
            return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        pred_edge = sobel(pred)
        gt_edge = sobel(gt)
        return self.mse_loss(pred_edge, gt_edge)
    def total_variation_loss(self, x):
        diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return torch.mean(diff_h) + torch.mean(diff_v)
    def smooth_loss(self, x):
        diff_h2 = x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
        diff_v2 = x[:, :, 2:, :] - 2 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
        return torch.mean(torch.abs(diff_h2)) + torch.mean(torch.abs(diff_v2))
    def forward(self, hr_pred, hr_gt, lr_center=None, level=None, width=None,
                deg_z_lr=None, deg_z_pred=None):
        pixel_loss = self.l1_loss(hr_pred, hr_gt)
        total_loss = pixel_loss
        if self.use_edge_loss:
            edge_loss = self.gradient_loss(hr_pred, hr_gt)
            total_loss += self.edge_weight * edge_loss
        if self.use_tv_loss:
            tv_loss = self.total_variation_loss(hr_pred)
            total_loss += self.tv_weight * tv_loss
        if self.use_smooth_loss:
            smooth_loss = self.smooth_loss(hr_pred)
            total_loss += self.smooth_weight * smooth_loss
        # HU 保真损失：反窗宽窗位到 HU 空间做 L1（临床意义：绝对 CT 值可解释）
        if self.use_hu_loss and level is not None and width is not None:
            level = torch.as_tensor(level, device=hr_pred.device, dtype=hr_pred.dtype)
            width = torch.as_tensor(width, device=hr_pred.device, dtype=hr_pred.dtype)
            low = (level - width / 2.0).view(-1, 1, 1, 1)
            span = (width).view(-1, 1, 1, 1)
            pred_hu = low + (hr_pred * 0.5 + 0.5) * span
            gt_hu = low + (hr_gt * 0.5 + 0.5) * span
            total_loss += self.hu_weight * self.l1_loss(pred_hu, gt_hu)
        # 下采样一致性：HR 预测下采样后应与观测 LR（中心通道）一致
        if self.use_consist_loss and lr_center is not None:
            lr_hat = F.interpolate(hr_pred, scale_factor=1.0 / self.upscale,
                                   mode='bicubic', align_corners=False)
            total_loss += self.consist_weight * self.l1_loss(lr_hat, lr_center)
        # 退化码一致性：下采样 HR 的退化表征应与观测 LR 退化表征一致
        if self.use_deg_loss and deg_z_lr is not None and deg_z_pred is not None:
            total_loss += self.deg_weight * self.l1_loss(deg_z_lr, deg_z_pred)
        return total_loss
