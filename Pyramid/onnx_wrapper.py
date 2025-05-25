import torch
import torch.nn as nn

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs_0, obs_1, obs_2, obs_3, action_masks):
        # obs_0: [batch, 56], obs_1: [batch, 56], obs_2: [batch, 56], obs_3: [batch, 4]
        x = torch.cat([obs_0, obs_1, obs_2, obs_3], dim=1)  # [batch, 172]
        logits = self.model(x)  # [batch, 5]
        # Apply action mask: set logits of masked actions to a large negative value
        # action_masks: [batch, 5], 1 for valid, 0 for invalid
        mask = (action_masks == 0)
        logits = logits.masked_fill(mask, -1e9)
        action = torch.argmax(logits, dim=1, keepdim=True)  # [batch, 1]
        return action.to(torch.int64)