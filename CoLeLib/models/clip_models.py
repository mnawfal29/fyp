import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

#############################################
### CLIP Text Encoder Parameter-Efficient ###
#############################################


class CLIPTextModelForPromptTuning(nn.Module):
    def __init__(
        self, model: object, D_g: int, D_s: int, deep_replace_method: str = "replace"
    ):
        """
        CLIP Text Encoder for PE
        model: CLIP Text Encoder
        D_g_l: number of layers to append general prompts
        D_s: number of layers to append shared prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        """
        super().__init__()
        self.model = model
        self.d_model = 512
        self.D_g = D_g
        self.D_s = D_s
        self.deep_replace_method = deep_replace_method

    def forward(
        self,
        text_tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        g_prompt: torch.Tensor,
        s_prompt: torch.Tensor,
    ):
        """
        text_tokens: [batch_size, n_tokens]
        attn_mask: [batch_size, n_tokens]
        g_prompt: [text_tokens, n_deep, n_prompts, d_model]
        s_prompt: [text_tokens, n_deep, n_prompts, d_model]
        """
        assert g_prompt.size(0) == s_prompt.size(0)
        bs = g_prompt.size(0)

        g_prompt = g_prompt.to(text_tokens.device)
        s_prompt = s_prompt.to(text_tokens.device)

        x = self.model.embeddings.token_embedding(text_tokens)
        p = g_prompt[:, 0]
        L_p = p.size(1)

        x = torch.cat([x[:, 0:1, :], p, x[:, 1:, :]], dim=1)
        x = x + self.model.embeddings.position_embedding(
            torch.arange(x.size(1), device=attn_mask.device).unsqueeze(0)
        )

        total_num_layers = len(self.model.encoder.layers)

        for i, l in enumerate(self.model.encoder.layers):
            if i > 0:
                if i < self.D_g:
                    p = g_prompt[:, i]
                elif i >= (total_num_layers - self.D_s):
                    p = s_prompt[:, i - (total_num_layers - self.D_s)]

                if self.deep_replace_method == "accumulate":
                    previous_p_out = x[:, 1 : (L_p + 1), :]
                    p = torch.cat([previous_p_out, p], dim=1)
                x = torch.cat([x[:, 0:1, :], p, x[:, (L_p + 1) :, :]], dim=1)
                L_p = p.size(1)

            attn_mask_ = torch.cat(
                [torch.ones(bs, L_p, device=attn_mask.device), attn_mask], dim=-1
            )

            res = x
            x = l.layer_norm1(x)

            q = l.self_attn.q_proj(x) * 0.125
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            extended_attn_mask = (attn_mask_.unsqueeze(1).unsqueeze(1) == 0).float()
            extended_attn_mask[extended_attn_mask == 1] = torch.finfo(x.dtype).min

            q = q.view(x.size(0), x.size(1), 8, -1).transpose(1, 2)
            k = k.view(x.size(0), x.size(1), 8, -1).transpose(1, 2)
            v = v.view(x.size(0), x.size(1), 8, -1).transpose(1, 2)
            w = q @ k.transpose(-1, -2)
            
            c_mask = (
                _create_4d_causal_attention_mask((x.size(0), x.size(1)), torch.float, attn_mask.device)
            )
            w = w + c_mask + extended_attn_mask
            w = w.softmax(dim=-1)
            v = (w @ v).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
            x = l.self_attn.out_proj(v)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)

            x = res + x

        x = self.model.final_layer_norm(x)

        index = text_tokens.argmax(dim=-1) + L_p
        return x[torch.arange(x.size(0)), index]


###############################################
### CLIP Vision Encoder Parameter-Efficient ###
###############################################


class CLIPVisionModelForPromptTuning(nn.Module):
    def __init__(
        self, model: object, D_g: int, D_s: int, deep_replace_method: str = "replace"
    ):
        """
        CLIP Vision Encoder for PE
        model: CLIP Vision Encoder
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        """
        super().__init__()
        self.model = model
        self.d_model = 768
        self.D_g = D_g
        self.D_s = D_s
        self.deep_replace_method = deep_replace_method

    def forward(self, image: torch.Tensor, g_prompt: torch.Tensor, s_prompt: torch.Tensor):
        """
        image: [batch_size, 3, 224, 224]
        g_prompt: [batch_size, n_deep, n_prompts, d_model]
        """
        x = self.model.embeddings(image)

        p = g_prompt[:, 0]

        x = torch.cat([x, p], dim=1)
        x = self.model.pre_layrnorm(x)
        L_p = p.size(1)

        total_num_layers = len(self.model.encoder.layers)

        for i, l in enumerate(self.model.encoder.layers):
            if i > 0:
                if i < self.D_g:
                    p = g_prompt[:, i]
                elif i >= (total_num_layers - self.D_s):
                    p = s_prompt[:, i - (total_num_layers - self.D_s)]

                if self.deep_replace_method == "accumulate":
                    previous_p_out = x[:, -L_p:, :]
                    p = torch.cat([previous_p_out, p], dim=1)
                x = torch.cat([x[:, :-L_p, :], p], dim=1)
                L_p = p.size(1)


            res = x
            x = l.layer_norm1(x)

            q = l.self_attn.q_proj(x) * 0.125
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            q = q.view(x.size(0), x.size(1), 12, -1).transpose(1, 2)
            k = k.view(x.size(0), x.size(1), 12, -1).transpose(1, 2)
            v = v.view(x.size(0), x.size(1), 12, -1).transpose(1, 2)
            w = q @ k.transpose(-1, -2)
            w = w.softmax(dim=-1)
            v = (w @ v).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
            x = l.self_attn.out_proj(v)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)

            x = res + x

        return self.model.post_layernorm(x[:, 0, :])


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.qkv_proj = nn.Linear(input_dim, embed_dim * 3)  # Single projection for Q, K, V
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_length, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each shape: (batch_size, seq_length, num_heads, head_dim)
        
        # Transpose for multi-head processing: (batch_size, num_heads, seq_length, head_dim)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        
        return self.out_proj(out)  # Final linear projection
    

################################
### CLIP Parameter-Efficient ###
################################


class CLIPParameterEfficient(nn.Module):
    def __init__(
        self,
        L_g: int = 2,
        L_s: int = 2,
        D_g: int = 3,
        D_s: int = 3,
        text_deep_replace_method: str = "replace",
        vision_deep_replace_method: str = "replace",
    ):
        """
        CLIP Parameter-Efficient
        L_g: number of g-prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        """
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        ### Text Encoder ###
        self.text_model = CLIPTextModelForPromptTuning(
            model=self.clip_model.text_model,
            D_g=D_g,
            D_s=D_s,
            deep_replace_method=text_deep_replace_method,
        )

        ### Vision Encoder ###
        self.vision_model = CLIPVisionModelForPromptTuning(
            model=self.clip_model.vision_model,
            D_g=D_g,
            D_s=D_s,
            deep_replace_method=vision_deep_replace_method,
        )

        self.image_proj = self.clip_model.visual_projection
        self.text_proj = self.clip_model.text_projection
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        self.prompt_proj = MultiHeadSelfAttention(self.text_model.d_model, self.vision_model.d_model)
        # self.prompt_proj = nn.Linear(self.text_model.d_model, self.vision_model.d_model)
        self.g_v_values = nn.Parameter(torch.zeros(D_g, L_g, self.vision_model.d_model))
        self.g_l_values = nn.Parameter(torch.zeros(D_g, L_g, self.text_model.d_model))
        self.s_values = nn.Parameter(torch.zeros(D_s, L_s, self.text_model.d_model))

        nn.init.xavier_uniform_(self.s_values.data)

        self.L_g = L_g
        self.L_s = L_s
        self.D_g = D_g
        self.D_s = D_s

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        device="cuda",
    ):
        """
        image: [batch_size, 3, 224, 224]
        text_tokens: [n_classes, max_length]
        attn_mask: [n_classes, max_length]
        """
        batch_size = image.shape[0]

        text_g_prompt = self.g_l_values.repeat(text_tokens.size(0), 1, 1, 1).to(device)
        vision_g_prompt = self.g_v_values.repeat(batch_size, 1, 1, 1)
        text_s_prompt = self.s_values.repeat(text_tokens.size(0), 1, 1, 1).to(device)
        vision_s_prompt = self.prompt_proj(self.s_values).repeat(batch_size, 1, 1, 1)

        text_out = self.text_model(text_tokens, attn_mask, text_g_prompt, text_s_prompt)
        img_out = self.vision_model(image, vision_g_prompt, vision_s_prompt)

        # Project to common dimensional space
        text_proj = self.text_proj(text_out)
        img_proj = self.image_proj(img_out)
        # Normalize
        text_embed = text_proj / text_proj.norm(dim=-1, keepdim=True)
        img_embed = img_proj / img_proj.norm(dim=-1, keepdim=True)
        sim = 100 * img_embed @ text_embed.T

        return sim
