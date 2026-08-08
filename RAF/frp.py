import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionPromptLearner(nn.Module):
    def __init__(self, channels=2048, spatial_size=7, temperature=0.1):
        super(RegionPromptLearner, self).__init__()

        self.temperature = temperature
        self.channels = channels
        feature_path = "/root/pre_back_features.pt"

        # f_random
        self.learnable_prompts = nn.Parameter(torch.randn(8, channels, spatial_size, spatial_size))
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        # f_b
        # dim 512 2048 7 7
        self.background_proxies = torch.load(feature_path, map_location='cpu')
        self.background_proxies.requires_grad_(False)

    def forward(self, fr):
        """
        Args:
            fr: Region proposal features, shape (8, 2048, 7, 7)
        Returns:
            loss: Contrastive loss value
            f_rp: Processed prompts (optional, for visualization or fusion)
        """
        # Step 1: 生成 f_rp
        f_rp = self.prompt_encoder(self.learnable_prompts)  # Shape: (8, 2048, 7, 7)


        f_rp_norm = F.normalize(f_rp, p=2, dim=1)
        fr_norm = F.normalize(fr, p=2, dim=1)
        fb_norm = F.normalize(self.background_proxies, p=2, dim=1)  # Shape: (512, 2048, 7, 7)


        B, C, H, W = f_rp_norm.shape
        S = H * W

        f_rp_flat = f_rp_norm.view(B, C, S).permute(0, 2, 1).contiguous()  # (8, 49, 2048)
        fr_flat = fr_norm.view(B, C, S).permute(0, 2, 1).contiguous()  # (8, 49, 2048)
        fb_flat = fb_norm.view(-1, C, S).permute(0, 2, 1).contiguous()  # (512, 49, 2048)

        # --- 计算 Positive Score (Pull fr) ---
        # Dot product between f_rp and fr
        # (8, 49, 2048) x (8, 2048, 49) -> (8, 49, 49)
        # 注意：通常对比学习是在同一空间位置比较，所以我们对 S 维度做逐元素乘积求和
        pos_sim = torch.sum(f_rp_flat * fr_flat, dim=-1) / self.temperature  # (8, 49)

        # --- 计算 Negative Score (Push fb) ---
        # f_rp (8, 49, 2048) vs fb (512, 49, 2048)
        # 需要广播机制。
        # unsqueeze fb to (1, 512, 49, 2048), f_rp to (8, 1, 49, 2048)
        f_rp_exp = f_rp_flat.unsqueeze(1)  # (8, 1, 49, 2048)
        fb_exp = fb_flat.unsqueeze(0)  # (1, 512, 49, 2048)

        # Sum over Channel dim (2048)
        neg_sim = torch.sum(f_rp_exp * fb_exp, dim=-1) / self.temperature  # (8, 512, 49)

        # Step 4: 计算 InfoNCE Loss
        # LogSumExp trick for numerical stability
        # Concatenate positive score (unsqueeze to match negative dim) and negative scores
        # pos_sim: (8, 49) -> (8, 1, 49)
        # neg_sim: (8, 512, 49)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (8, 513, 49)

        # Labels are all 0 (indicating the first element is the positive pair)
        labels = torch.zeros(B, S, dtype=torch.long, device=logits.device)

        # CrossEntropyLoss applies Softmax and Log automatically
        # We need to transpose logits to (N, C, S) format? No, standard CE expects (N, C) or (N, C, ...)
        # Here our "Classes" are the 513 candidates (1 pos + 512 neg)
        # Shape required: (Batch, Num_Classes, Spatial) -> (8, 513, 49)
        # Target Shape: (Batch, Spatial) -> (8, 49)
        loss = F.cross_entropy(logits, labels)

        return loss, f_rp



def train_step():
    fr_input = torch.load("/root/region_proposal_features.pt").cuda()  # Shape: (8, 2048, 7, 7)
    # 实例化模型
    model = RegionPromptLearner(
        channels=2048,
        spatial_size=7,
        temperature=0.1
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.1
    )
    model.train()
    for i in range(20):
        optimizer.zero_grad()
        loss, _ = model(fr_input)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    if torch.cuda.is_available():
        train_step()
    else:
        print("CUDA not available, running on CPU (will be slow for 2048 dims)")
        pass
