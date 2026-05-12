import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp
from functools import partial


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        t = t * 1000
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



def normalize_adj(adj: torch.Tensor, eps=1e-6):
    # adj: [P,P], 1=allow, 0=block. Make symmetric, add self loops, D^{-1/2} A D^{-1/2}
    adj = adj.float()
    adj = ((adj + adj.T) > 0).float()
    adj.fill_diagonal_(1.0)
    d = adj.sum(-1).clamp_min(eps)
    d_inv_sqrt = d.pow(-0.5)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

class PathwayGraphBlock(nn.Module):
    def __init__(self, hidden_dim, pathway_adj, num_heads=4, drop=0.1):
        super().__init__()
        adj_norm = normalize_adj(pathway_adj)
        attn_mask = (adj_norm == 0)                 # True = mask
        self.register_buffer("attn_mask", attn_mask)  # [P,P]
        add_mask = torch.zeros_like(adj_norm, dtype=torch.float32)
        add_mask = add_mask.masked_fill(attn_mask, -1e4)
        self.register_buffer("add_mask", add_mask)

        self.norm1 = nn.LayerNorm(hidden_dim)
        # Use PyTorch MultiheadAttention to support attention masks
        self.attn  = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, bias=True)
        self.drop  = nn.Dropout(drop)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable residual scale

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):  # x: [B,P,H]
        B, P, _ = x.shape
        # attention with graph mask; MultiheadAttention expects [P,P] or [B*H,P,P] attn_mask
        # Convert boolean mask (True=block) to additive mask with -inf on blocked positions
        qkv = self.norm1(x)
        y, _ = self.attn(qkv, qkv, qkv, attn_mask=self.add_mask)
        x = x + self.drop(self.alpha * y)           # residual
        x = x + self.mlp(self.norm2(x))             # FFN
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    pathway_size: List[int] (num_pathways,)
    """

    def __init__(self, hidden_size, pathway_size):
        super().__init__()
        self.num_pathways = len(pathway_size)
        self.norm_final = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.adaLN_modulation = nn.ModuleList()
        for out_channels in pathway_size:
            self.norm_final.append(nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6))
            self.linear.append(nn.Linear(hidden_size, out_channels, bias=True))
            self.adaLN_modulation.append(nn.Sequential(
                nn.SiLU(), 
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            ))


    def forward(self, x, c):
        # x: [B, num_pathways, hidden_size]
        x_list = []
        for i in range(self.num_pathways):
            xi = x[:, i:i+1, :]  # [B, 1, hidden_size]
            shift, scale = self.adaLN_modulation[i](c).chunk(2, dim=1)
            xi = modulate(self.norm_final[i](xi), shift, scale)
            xi = self.linear[i](xi)  # [B, pathway_size[i]]
            x_list.append(xi.squeeze(1))
        return x_list


class Block(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


    
class GenePathwayEmbedding(nn.Module):
    def __init__(self, gene_dim, hidden_dim, pathway_indices=None, num_pathway=None, pathway_size=None, bg_gene_indices=None):
        """
        gene_dim: number of genes in input
        hidden_dim: num hidden dimension
        pathway_indices: List[torch.Tensor(int)] (num_pathway)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bg_gene_indices = bg_gene_indices
        self.pathway_indices = pathway_indices
        self.num_pathway = num_pathway
        self.pathway_size = pathway_size

        self.gene_pathway_ebd = nn.Parameter(
            torch.randn(self.num_pathway, hidden_dim) / (hidden_dim ** 0.5),
            requires_grad=True) 

        self.gene_count_ebd = nn.ModuleList()
        for pathway_idx in range(self.num_pathway):
            self.gene_count_ebd.append(nn.Sequential(
                nn.Linear(self.pathway_size[pathway_idx], hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
            ))
        self.bg_gene_count_ebd = nn.Sequential(
            nn.Linear(len(self.bg_gene_indices), hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.bg_gene_bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))


    def forward(self, x):
        """
        x: (N, NumGene) tensor of inputs

        """
        x = x.squeeze(1)  # (N, NumGene)
        B, L = x.shape
        gene_count_ebd_list = []
        for i in range(self.num_pathway):
            gene_indices = self.pathway_indices[i]
            gene_in_pathway = x[:, gene_indices] # B, pathway_size
            gene_count_ebd = self.gene_count_ebd[i](gene_in_pathway) + self.gene_pathway_ebd[i] # B, hidden_dim
            gene_count_ebd_list.append(gene_count_ebd) # gene_count_ebd_list: B, num_pathway, hidden_dim
        gene_pathway_count_ebd = torch.stack(gene_count_ebd_list, dim=1) # B, num_pathway, hidden_dim

        bg_gene_count_ebd = self.bg_gene_count_ebd(x[:, self.bg_gene_indices]).unsqueeze(1) + self.bg_gene_bias # B, 1, hidden_dim

        return gene_pathway_count_ebd, bg_gene_count_ebd # B, num_pathway, hidden_dim, B, 1, hidden_dim


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads = heads),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class BGHead(nn.Module):
    def __init__(self, hidden_dim, bg_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        self.fc  = nn.Linear(hidden_dim, bg_size)

    def forward(self, bg_tok, c):
        # bg_tok: [B,1,H], c: [B,H]
        shift, scale = self.ada(c).chunk(2, dim=-1)
        x = self.norm(bg_tok)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.fc(x)  # [B,1,bg_size]
        return out.squeeze(1)  # [B,bg_size]

class RNAFMModel(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        num_clusters=100,
        gene_dim=20820,
        hidden_dim=512,
        num_head=8,
        attn_drop=0.0,
        proj_drop=0.0,
        depth=12,
        cond_drop_ratio=0.,
        pathway_indices=None, # (num_pathways, pathway_size)
        pathway_adj=None,
        bg_gene=None
    ):
        super().__init__()

        self.gene_dim = gene_dim
        # self.condition_dim = condition_dim
        self.cond_drop_ratio = cond_drop_ratio
        self.num_pathways = len(pathway_indices)
        self.pathway_adj = pathway_adj
        # make bg_gene in one tensor
        self.bg_gene = bg_gene # (num_bg_gene,)
        self.bg_gene_size = len(self.bg_gene)
        print(f'num_bg_gene: {self.bg_gene_size}')
        
        indices_list = []
        for idx in pathway_indices:
            if isinstance(idx, torch.Tensor):
                indices_list.append(idx.long())
            else:
                indices_list.append(torch.tensor(idx, dtype=torch.long))

        self.pathway_indices = indices_list
        self.pathway_size = [len(p) for p in self.pathway_indices]
        print(f'num_pathways: {self.num_pathways}')
        print(f'pathway_size: {self.pathway_size}')
        
        empty_cond = torch.randn(1, num_clusters, input_dim) / (input_dim ** 0.5)
        self.register_buffer("empty_cond", empty_cond)

        # input layers
        self.gene_embedder = GenePathwayEmbedding(gene_dim, hidden_dim, pathway_indices=self.pathway_indices, num_pathway=self.num_pathways, pathway_size=self.pathway_size, bg_gene_indices=self.bg_gene)
        self.pathway_graph = PathwayGraphBlock(hidden_dim, self.pathway_adj)
        self.bg_head = BGHead(hidden_dim, self.bg_gene_size)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.image_embedder = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            nn.SiLU(),
            nn.Linear(input_dim, hidden_dim, bias=True),
        )
        self.cluster_index_emb = nn.Parameter(torch.randn(num_clusters, input_dim) / (input_dim ** 0.5))
        self.transformer = Transformer(input_dim, 2, num_head, hidden_dim//num_head, hidden_dim*4)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # transformer layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim,
                    num_heads=num_head,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        # output layers
        self.final_layer = FinalLayer(hidden_dim, self.pathway_size)
        self.initialize()

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.image_embedder[0].weight, std=0.02)
        nn.init.normal_(self.image_embedder[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        for i in range(self.num_pathways):
            nn.init.constant_(self.final_layer.adaLN_modulation[i][-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[i][-1].bias, 0)
            nn.init.constant_(self.final_layer.linear[i].weight, 0)
            nn.init.constant_(self.final_layer.linear[i].bias, 0)
        
    def forward(
        self,
        x,
        t,
        y
    ):
        """
        x: B, gene_dim
        t: B, 1
        y: B, num_clusters, input_dim
        """
        B = x.shape[0]
                
        x, bg_gene_count_ebd = self.gene_embedder(x) # B, num_pathways, hidden_dim, B, 1, hidden_dim
        x = self.pathway_graph(x)

            
        t = self.t_embedder(t) # B, hidden_dim

        if self.training and torch.rand(1) < self.cond_drop_ratio:
            y = self.empty_cond.repeat(B, 1, 1)

        y = y + self.cluster_index_emb # B, num_clusters, input_dim

        y = self.transformer(y) # B, num_clusters, input_dim
            
        y = y.mean(dim = 1) # B, input_dim
        y = self.image_embedder(y) # B, hidden_dim
        c = t + y


        for i, block in enumerate(self.blocks):
            x = block(x, c) # B, num_pathways, hidden_dim
        x_list = self.final_layer(x, c) # list: len=num_pathways, each [B, pathway_size[i]]
        bg_pred = self.bg_head(bg_gene_count_ebd, c) # B, bg_gene_size
        count_x = torch.zeros(B, self.gene_dim, device=x.device, dtype=x_list[0].dtype)
        final_x = torch.zeros(B, self.gene_dim, device=x.device, dtype=x_list[0].dtype)
        for i in range(self.num_pathways):
            # scatter the model_out[:,i,:] to the final_x at the corresponding pathway indices, average the overlaps
            idx = self.pathway_indices[i].unsqueeze(0).repeat(B, 1)
            count_x.scatter_add_(dim=1, index=idx, src=torch.ones_like(x_list[i]))
            final_x.scatter_add_(dim=1, index=idx, src=x_list[i])
        bg_idx = self.bg_gene.unsqueeze(0).repeat(B, 1)
        count_x.scatter_add_(dim=1, index=bg_idx, src=torch.ones_like(bg_pred))
        final_x.scatter_add_(dim=1, index=bg_idx, src=bg_pred)

        assert count_x.min() > 0
        final_x.div_(count_x)
        
        return final_x
        

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        
        Args:
            x: [B, gene_dim] - gene expression
            t: [B] - timestep
            y: [B, num_clusters, input_dim] - cluster features
            cfg_scale: float - guidance scale (1.0 = no guidance, >1.0 = more guidance)
        """
        B = x.shape[0]
        assert B == y.shape[0] == t.shape[0]
        
        # Duplicate inputs for conditional and unconditional paths
        x_combined = torch.cat([x, x], dim=0)  # [2B, gene_dim]
        t_combined = torch.cat([t, t], dim=0)  # [2B]
        y_cond = y  # [B, num_clusters, input_dim]
        y_uncond = self.empty_cond.repeat(B, 1, 1)  # [B, 1, input_dim]
        
        # Combine conditions
        y_combined = torch.cat([y_cond, y_uncond], dim=0)  # [2B, num_clusters, input_dim]
        
        # Forward pass through model
        model_out = self.forward(x_combined, t_combined, y_combined)  # [2B, num_pathways, pathway_size]
        # Split into conditional and unconditional outputs
        cond_out = model_out[:B]  # [B, gene_dim] - conditional prediction
        uncond_out = model_out[B:]  # [B, gene_dim] - unconditional prediction
        
        # Apply classifier-free guidance
        # CFG formula: uncond + cfg_scale * (cond - uncond)
        guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return guided_out

