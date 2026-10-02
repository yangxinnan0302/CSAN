import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from collections import OrderedDict

from GAT import GATLayer
from pytorch_pretrained_bert.modeling import BertModel


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X."""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    return torch.div(X, norm)


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X."""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


def cross_attention(query, context, matrix, smooth, eps=1e-8):
    """
    Compute cross-attention weighted context.
    query:   (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    matrix = matrix.to(query.device)

    query = torch.mul(query, matrix)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d) x (batch, d, queryL) -> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, dim=-1)

    # (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = F.softmax(attn * smooth, dim=2)

    # (batch, sourceL, queryL) -> weighted context
    attnT = torch.transpose(attn, 1, 2).contiguous()
    contextT = torch.transpose(context, 1, 2)
    wcontext = torch.bmm(contextT, attnT)           # (batch, d, queryL)
    wcontext = torch.transpose(wcontext, 1, 2)      # (batch, queryL, d)
    wcontext = l2norm(wcontext, dim=-1)
    return wcontext


# ---------------------------------------------------------------------------
# Encoder Modules
# ---------------------------------------------------------------------------

class EncoderImage(nn.Module):
    """
    Build local region representations via a single FC layer.
    Input:  images  (batch_size, 36, img_dim)
    Output: img_emb (batch_size, 36, embed_size)
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super().__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self._init_weights()

    def _init_weights(self):
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        img_emb = self.fc(images)
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)
        return img_emb

    def load_state_dict(self, state_dict):
        """Accept state_dict from a full model checkpoint."""
        own_state = self.state_dict()
        new_state = OrderedDict(
            {k: v for k, v in state_dict.items() if k in own_state}
        )
        super().load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    BERT-based text encoder with an optional fine-tuning flag.
    """

    def __init__(self, opt):
        super().__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        if not opt.ft_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('text-encoder-bert: frozen (no grad)')
        else:
            print('text-encoder-bert: fine-tuning enabled')

        self.embed_size = opt.embed_size
        self.fc = nn.Sequential(
            nn.Linear(opt.bert_size, opt.embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, captions, lengths):
        all_encoders, _ = self.bert(captions)
        out = all_encoders[-1]
        out = self.fc(out)
        return out


# ---------------------------------------------------------------------------
# Self-Attention Modules
# ---------------------------------------------------------------------------

class VisualSA(nn.Module):
    """
    Build global image representation by self-attention over local regions.
    Input:  local      (batch_size, 36, embed_dim)
            raw_global (batch_size, embed_dim)
    Output: new_global (batch_size, embed_dim)
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super().__init__()
        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_region),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.softmax(self.embedding_common(common).squeeze(2))

        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)
        return new_global


class TextSA(nn.Module):
    """
    Build global text representation by self-attention over local words.
    Input:  local      (batch_size, L, embed_dim)
            raw_global (batch_size, embed_dim)
    Output: new_global (batch_size, embed_dim)
    """

    def __init__(self, embed_dim, dropout_rate):
        super().__init__()
        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, local, raw_global):
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.softmax(self.embedding_common(common).squeeze(2))

        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)
        return new_global


class SCAN_attention(nn.Module):
    """
    SCAN-style cross-attention.
    query:   (n_context, queryL,  d)
    context: (n_context, sourceL, d)
    """

    def __init__(self, embed_size):
        super().__init__()
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)

    def forward(self, query, context, smooth, eps=1e-8):
        sim_k = self.fck(context)
        sim_q = self.fcq(query)
        sim_v = context

        attn = torch.bmm(sim_k, sim_q.permute(0, 2, 1))
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
        attn = F.softmax(attn.permute(0, 2, 1) * smooth, dim=2)

        weighted_context = torch.bmm(attn, sim_v)
        weighted_context = l2norm(weighted_context, dim=-1)
        return weighted_context


# ---------------------------------------------------------------------------
# Graph Modules
# ---------------------------------------------------------------------------

class GraphReasoning(nn.Module):
    """
    Similarity graph reasoning on a fully-connected graph.
    Input:  sim_emb  (batch_size, L+1, sim_dim)
    Output: sim_sgr  (batch_size, L+1, sim_dim)
    """

    def __init__(self, sim_dim):
        super().__init__()
        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w   = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w   = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key   = self.graph_key_w(sim_emb)
        sim_edge  = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr   = torch.bmm(sim_edge, sim_emb)
        sim_sgr   = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr


class AttentionFiltration(nn.Module):
    """
    Gate-based attention filtration.
    Input:  sim_emb (batch_size, L+1, sim_dim)
    Output: sim_saf (batch_size, sim_dim)
    """

    def __init__(self, sim_dim):
        super().__init__()
        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, sim_emb):
        sim_attn = l1norm(
            torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))),
            dim=-1,
        )
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf


# ---------------------------------------------------------------------------
# Regulator Modules
# ---------------------------------------------------------------------------

class Aggregation_regulator(nn.Module):
    """RAR: aggregation regulator that re-weights mid-level representations."""

    def __init__(self, sim_dim, embed_dim):
        super().__init__()
        self.rar_q_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim), nn.Tanh(), nn.Dropout(0.4)
        )
        self.rar_k_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim), nn.Tanh(), nn.Dropout(0.4)
        )
        self.rar_v_w = nn.Sequential(nn.Linear(sim_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mid, hig):
        mid_k = self.rar_k_w(mid)
        hig_q = self.rar_q_w(hig).unsqueeze(1).repeat(1, mid_k.size(1), 1)

        weights = self.softmax(self.rar_v_w(mid_k.mul(hig_q)).squeeze(2))
        new_hig = (weights.unsqueeze(2) * mid).sum(dim=1)
        new_hig = l2norm(new_hig, dim=-1)
        return new_hig


class Correpondence_regulator(nn.Module):
    """RCR: correspondence regulator that updates alignment matrix and smooth."""

    def __init__(self, sim_dim, embed_dim):
        super().__init__()
        self.rcr_smooth_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim // 2),
            nn.Tanh(),
            nn.Linear(sim_dim // 2, 1),
        )
        self.rcr_matrix_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim * 2),
            nn.Tanh(),
            nn.Linear(sim_dim * 2, embed_dim),
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, matrix, smooth):
        matrix = matrix.to(x.device)
        matrix = (self.tanh(self.rcr_matrix_w(x)) + matrix).clamp(min=-1, max=1)
        smooth = self.relu(self.rcr_smooth_w(x) + smooth)
        return matrix, smooth


class Alignment_vector(nn.Module):
    """Compute alignment vector between query and context."""

    def __init__(self, sim_dim, embed_dim):
        super().__init__()
        self.sim_transform_w = nn.Linear(embed_dim, sim_dim)

    def forward(self, query, context, matrix, smooth):
        wcontext = cross_attention(query, context, matrix, smooth)
        sim_rep  = torch.pow(torch.sub(query, wcontext), 2)
        sim_rep  = l2norm(self.sim_transform_w(sim_rep), dim=-1)
        return sim_rep


# ---------------------------------------------------------------------------
# ADAPT Module
# ---------------------------------------------------------------------------

class ADAPT(nn.Module):
    """
    Adaptive feature transformation conditioned on a query vector.
    Applies learned gamma/beta shifts (FiLM-style) to value features.
    """

    def __init__(
        self,
        k=None,
        q1_size=None,
        q2_size=None,
        v1_size=None,
        v2_size=None,
        nonlinear_proj=False,
        groups=1,
        sg_dim=None,
    ):
        super().__init__()
        self.groups = groups

        if nonlinear_proj:
            self.fc_gamma = nn.Sequential(
                nn.Linear(q1_size, v1_size),
                nn.ReLU(inplace=True),
                nn.Linear(q1_size, v1_size),
            )
            self.fc_beta = nn.Sequential(
                nn.Linear(q2_size, v2_size),
                nn.ReLU(inplace=True),
                nn.Linear(q2_size, v2_size),
            )
        else:
            print("Initializing linear ADAPT")
            if q1_size != v1_size:
                self.v1_transform = nn.Sequential(nn.Linear(v1_size, q1_size))
                v1_size = q1_size
            self.fc_gamma = nn.Sequential(nn.Linear(q1_size, v1_size // groups))
            self.fc_beta  = nn.Sequential(nn.Linear(q1_size, v1_size // groups))

        # Visualization hooks are opt-in and therefore add no tensor-copying
        # overhead to normal training or evaluation.
        self.record_modulation = False
        self.last_input = None
        self.last_output = None
        self.last_delta = None
        self.last_gamma = None
        self.last_beta = None

    def forward(self, value1, value2, query1, query2):
        B,  D,  rk = value1.shape
        Bv, Dv     = query1.shape

        if D != Dv:
            value1 = self.v1_transform(value1.permute(0, 2, 1)).permute(0, 2, 1)

        value1 = value1.view(B, Dv // self.groups, self.groups, -1)

        gammas = self.fc_gamma(query1).view(Bv, Dv // self.groups, 1, 1)
        betas  = self.fc_beta(query1).view(Bv, Dv // self.groups, 1, 1)

        # All three sg_type branches produce the same result; unified here.
        normalized = value1 * (gammas + 1) + betas
        normalized = normalized.view(B, Dv, -1)

        if self.record_modulation:
            value1_flat = value1.view(B, Dv, -1)
            self.last_input = value1_flat.detach()
            self.last_output = normalized.detach()
            self.last_delta = (normalized - value1_flat).detach()
            self.last_gamma = gammas.detach()
            self.last_beta = betas.detach()
        return normalized


# ---------------------------------------------------------------------------
# GAT Modules
# ---------------------------------------------------------------------------

class GATopt:
    """Configuration object for GAT layers."""

    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_attention_heads          = 8
        self.hidden_dropout_prob          = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT1(nn.Module):
    """Stack of GATLayer modules (self-attention variant)."""

    def __init__(self, config_gat):
        super().__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config_gat.num_layers)]
        )

    def forward(self, querys, keys, values, attention_mask=None, position_weight=None):
        hidden_states = querys
        for layer_module in self.encoder:
            hidden_states = layer_module(
                hidden_states, hidden_states, hidden_states,
                attention_mask, position_weight,
            )
        return hidden_states  # (B, seq_len, D)


class GAT_111(nn.Module):
    """
    Dual-stream GAT block for image and text features with
    weighted residual connections and LayerNorm.
    """

    def __init__(self, opt):
        super().__init__()
        config_img = GATopt(opt.embed_size, 1)
        config_cap = GATopt(opt.embed_size, 1)
        self.gat_img = GAT1(config_img)
        self.gat_cap = GAT1(config_cap)

        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        # Learnable residual weights
        self.alpha_img = nn.Parameter(torch.tensor(0.5))
        self.alpha_cap = nn.Parameter(torch.tensor(0.5))

        # Projection layers
        self.img_linear = nn.Linear(opt.embed_size, opt.embed_size)
        self.cap_linear  = nn.Linear(opt.embed_size, opt.embed_size)

        # LayerNorm
        self.img_ln = nn.LayerNorm(opt.embed_size)
        self.cap_ln  = nn.LayerNorm(opt.embed_size)

    def forward(self, img_emb0, cap_emb0):
        # Image branch
        img_gat  = self.img_linear(self.gat_img(img_emb0, img_emb0, img_emb0))
        img_feat = self.img_ln(self.dropout(img_emb0 + self.alpha_img * img_gat))
        img_emb  = l2norm(img_feat, dim=-1)

        # Text branch
        cap_gat  = self.cap_linear(self.gat_cap(cap_emb0, cap_emb0, cap_emb0))
        cap_feat = self.cap_ln(self.dropout(cap_emb0 + self.alpha_cap * cap_gat))
        cap_emb  = l2norm(cap_feat, dim=-1)

        return img_emb, cap_emb


# ---------------------------------------------------------------------------
# Similarity Encoder
# ---------------------------------------------------------------------------

class EncoderSimilarity(nn.Module):
    """
    Compute image-text similarity via SGR or SAF with RCAR regulators.
    Input:  img_emb  (batch_size, 36, embed_size)
            cap_emb  (batch_size, L,  embed_size)
    Output: sim_all  (batch_size, batch_size)
    """

    def __init__(
        self,
        opt,
        embed_size,
        sim_dim,
        module_name='AVE',
        sgr_step=3,
        focal_type='equal',
        q1_size=1024,
        q2_size=1024,
        v1_size=1024,
        v2_size=1024,
        k=1,
    ):
        super().__init__()
        self.module_name = module_name
        self.opt       = opt
        self.embed_dim = embed_size
        self.sim_dim   = sim_dim

        # Global feature extractors
        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        # Projection layers
        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        self.sim_trancon_w = nn.Linear(embed_size, sim_dim)

        # Regulator step counts
        if opt.self_regulator == 'only_rar':
            rar_step, rcr_step, alv_step = opt.rar_step, 0, 1
        elif opt.self_regulator == 'only_rcr':
            rar_step, rcr_step, alv_step = 0, opt.rcr_step, opt.rcr_step
        elif opt.self_regulator == 'coop_rcar':
            rar_step, rcr_step, alv_step = opt.rcar_step, opt.rcar_step - 1, opt.rcar_step
        else:
            raise ValueError(f'Unknown opt.self_regulator: {opt.self_regulator}')

        self.fusion_gate = nn.Linear(opt.embed_size * 2, opt.embed_size)

        self.rar_modules = nn.ModuleList(
            [Aggregation_regulator(sim_dim, embed_size) for _ in range(rar_step)]
        )
        self.rcr_modules = nn.ModuleList(
            [Correpondence_regulator(sim_dim, embed_size) for _ in range(rcr_step)]
        )
        self.alv_modules = nn.ModuleList(
            [Alignment_vector(sim_dim, embed_size) for _ in range(alv_step)]
        )

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid    = nn.Sigmoid()
        self.focal_type = focal_type

        self.scan_attention = SCAN_attention(embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)

        self.cap_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        if module_name == 'SGR':
            self.SGR_module = nn.ModuleList(
                [GraphReasoning(sim_dim) for _ in range(sgr_step)]
            )
        elif module_name == 'SAF':
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError(f'Invalid opt.module_name: {module_name}')

        self.adapt_txt = ADAPT(k, v1_size, v2_size, q1_size, q2_size, nonlinear_proj=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, opt, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image   = img_emb.size(0)
        n_caption = cap_emb.size(0)

        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        smooth = opt.t2i_smooth if opt.attn_type == 't2i' else opt.i2t_smooth
        matrix = torch.ones(self.embed_dim)

        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i  = cap_emb[i, :n_word, :].unsqueeze(0)       # (1, L, D)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)          # (n_img, L, D)

            cap_guide        = cap_i.mean(dim=1)                 # (1, D)
            cap_guide_expand = cap_guide.repeat(n_image, 1)      # (n_img, D)

            # Text-guided image adaptation
            img_emb_adapted = self.adapt_txt(
                img_emb.permute(0, 2, 1),   # (n_img, D, num_boxes)
                img_glo,
                cap_guide_expand,
                None,
            )
            new_img_emb = img_emb_adapted.permute(0, 2, 1)     # (n_img, num_boxes, D)

            query   = cap_i_expand  if opt.attn_type == 't2i' else new_img_emb
            context = new_img_emb   if opt.attn_type == 't2i' else cap_i_expand

            for m, rar_module in enumerate(self.rar_modules):
                sim_mid = self.alv_modules[m](query, context, matrix, smooth)
                if m == 0:
                    sim_hig = torch.mean(sim_mid, 1)
                if m < (self.opt.rcar_step - 1):
                    matrix, smooth = self.rcr_modules[m](sim_mid, matrix, smooth)
                sim_hig = rar_module(sim_mid, sim_hig)

            sim_i = self.sigmoid(self.sim_eval_w(sim_hig))
            sim_all.append(sim_i)

        sim_all = torch.cat(sim_all, 1)
        return sim_all


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss."""

    def __init__(self, margin=0, max_violation=False):
        super().__init__()
        self.margin        = margin
        self.max_violation = max_violation
        self.CE = nn.CrossEntropyLoss()
        self.T  = 0.05

    def forward(self, scores):
        batch_size = scores.size(0)
        diagonal   = scores.diag().view(batch_size, 1)
        d1         = diagonal.expand_as(scores)
        d2         = diagonal.t().expand_as(scores)

        cost_s  = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(batch_size) > 0.5
        if torch.cuda.is_available():
            mask = mask.cuda()
        cost_s  = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s  = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        cost_s  = cost_s  / self.T
        cost_im = cost_im / self.T
        labels  = torch.arange(batch_size).long()
        if torch.cuda.is_available():
            labels = labels.cuda()

        return (self.CE(cost_im, labels) + self.CE(cost_s, labels)) / 2


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------

class CSAN(nn.Module):
    """Similarity Reasoning and Filtration (CSAN) Network."""

    def __init__(self, opt):
        super().__init__()
        self.grad_clip = opt.grad_clip

        self.img_enc   = EncoderImage(opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm)
        self.txt_enc   = EncoderText(opt)
        self.sim_enc   = EncoderSimilarity(
            opt, opt.embed_size, opt.sim_dim,
            opt.module_name, opt.sgr_step, opt.focal_type,
        )
        self.GAT_model = GAT_111(opt)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            self.GAT_model.cuda()

        self.criterion = ContrastiveLoss(margin=opt.margin, max_violation=opt.max_violation)

        # Separate parameter groups for BERT and other modules
        self.bert_params  = list(self.txt_enc.bert.parameters())
        self.other_params = (
            list(self.img_enc.parameters())
            + list(self.txt_enc.fc.parameters())
            + list(self.sim_enc.parameters())
            + list(self.GAT_model.parameters())
        )

        self.bert_optimizer  = torch.optim.Adam(self.bert_params,  lr=opt.bert_lr)
        self.other_optimizer = torch.optim.Adam(self.other_params, lr=opt.other_lr)
        self.optimizer = [self.bert_optimizer, self.other_optimizer]

        self.Eiters = 0

        # Cache used only by visualize_gafm.py.  It is deliberately excluded
        # from checkpoints and remains empty unless explicitly enabled.
        self._record_visualization = False
        self._visualization_cache = {}

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self):
        return [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
            self.sim_enc.state_dict(),
            self.GAT_model.state_dict(),
        ]

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])
        self.GAT_model.load_state_dict(state_dict[3])

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()
        self.GAT_model.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()
        self.GAT_model.eval()

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def enable_visualization(self, enabled=True):
        """Enable or disable lightweight GAFM interpretability hooks."""
        self._record_visualization = enabled
        self._visualization_cache = {}

        for gat_stream in (self.GAT_model.gat_img, self.GAT_model.gat_cap):
            for layer in gat_stream.encoder:
                layer.mha.record_attention = enabled
                layer.mha.last_attention_probs = None

        self.sim_enc.adapt_txt.record_modulation = enabled
        if not enabled:
            adapt = self.sim_enc.adapt_txt
            adapt.last_input = None
            adapt.last_output = None
            adapt.last_delta = None
            adapt.last_gamma = None
            adapt.last_beta = None

    def get_visualization_cache(self, cpu=True):
        """Return the tensors recorded by the most recent single-pair pass."""
        cache = dict(self._visualization_cache)

        img_mha = self.GAT_model.gat_img.encoder[-1].mha
        cap_mha = self.GAT_model.gat_cap.encoder[-1].mha
        if img_mha.last_attention_probs is not None:
            cache['ite_image_attention'] = img_mha.last_attention_probs
        if cap_mha.last_attention_probs is not None:
            cache['ite_text_attention'] = cap_mha.last_attention_probs

        adapt = self.sim_enc.adapt_txt
        for name in ('input', 'output', 'delta', 'gamma', 'beta'):
            value = getattr(adapt, 'last_' + name)
            if value is not None:
                cache['asm_' + name] = value

        if cpu:
            cache = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in cache.items()
            }
        return cache

    def forward_emb(self, images, captions, lengths):
        """Compute image and caption embeddings."""
        if torch.cuda.is_available():
            images   = images.cuda()
            captions = captions.cuda()

        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        if self._record_visualization:
            self._visualization_cache = {
                'image_before_ite': img_embs.detach(),
                'text_before_ite': cap_embs.detach(),
            }
        img_emb, cap_emb = self.GAT_model(img_embs, cap_embs)
        if self._record_visualization:
            self._visualization_cache.update({
                'image_after_ite': img_emb.detach(),
                'text_after_ite': cap_emb.detach(),
            })
        return img_emb, cap_emb, lengths

    def forward_sim(self, opt, img_embs, cap_embs, cap_lens):
        """Compute similarity scores."""
        scores = self.sim_enc(opt, img_embs, cap_embs, cap_lens)
        if self._record_visualization:
            self._visualization_cache['similarity_score'] = scores.detach()
        return scores

    def forward_loss(self, sims):
        """Compute contrastive loss."""
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_emb(self, opt, images, captions, lengths, ids=None, *args):
        """One training step."""
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('bert_lr',  self.optimizer[0].param_groups[0]['lr'])
        self.logger.update('other_lr', self.optimizer[1].param_groups[0]['lr'])

        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(opt, img_embs, cap_embs, cap_lens)

        for optimizer in self.optimizer:
            optimizer.zero_grad()

        loss = self.forward_loss(sims)
        loss.backward()

        for optimizer in self.optimizer:
            optimizer.step()
