import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, 3),
            Conv1dBlock(out_channels, out_channels, 3),
        ])
        
        # FiLM conditioning
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        scale, shift = embed.chunk(2, dim=1)
        out = out * (scale + 1) + shift
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim=4, local_cond_dim=None, global_cond_dim=256, diffusion_step_embed_dim=256, down_dims=[128, 256, 512]):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
        )
        
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim),
        ])

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # Symmetric to down loop (which makes only the final stage identity),
            # all up stages upsample. No is_last identity.
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim),
                Upsample1d(dim_in)
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=3),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond=None):
        # sample: (B, horizon, action_dim)
        # timestep: (B,)
        # global_cond: (B, global_cond_dim)
        sample = sample.transpose(1, 2) # (B, action_dim, horizon)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
            
        t_emb = self.time_mlp(timesteps)
        
        c = t_emb
        if global_cond is not None:
            c = torch.cat([c, global_cond], dim=-1)
            
        h = []
        x = sample
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, c)
            x = resnet2(x, c)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, c)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.transpose(1, 2) # (B, horizon, action_dim)
        return x
