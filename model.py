import torch
import torch.nn as nn
from GAT import GATLayer
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from collections import OrderedDict
import numpy as np
from collections import OrderedDict
import pickle
import copy
# import torchtext
from pytorch_pretrained_bert.modeling import BertModel


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        if not opt.ft_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('text-encoder-bert no grad')
        else:
            print('text-encoder-bert fine-tuning !')
        self.embed_size = opt.embed_size
        self.fc = nn.Sequential(nn.Linear(opt.bert_size, opt.embed_size), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, captions, lengths):
        all_encoders, pooled = self.bert(captions)
        out = all_encoders[-1]
        out = self.fc(out)
        return out


def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class IntrativeSelfattention(nn.Module):
    def __init__(self, embed_size, h, is_share, drop=None) -> None:
        super(IntrativeSelfattention, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear]
        else:
            # self.linears = clones(nn.Linear(embed_size, embed_size), 3)
            self.query = nn.Linear(embed_size, embed_size)
            self.key = nn.Linear(embed_size, embed_size)
            self.value = nn.Linear(embed_size, embed_size)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)

    def transpose_for_scores1(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        nbatches = q.size(0)
        nobjects = q.size(1)
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        query_head = self.transpose_for_scores1(query)
        # node_query = query_head.reshape(nbatches * self.h, query_head.size()[2], query_head.size()[3])
        key_head = self.transpose_for_scores1(key)
        # node_key = key_head.reshape(nbatches * self.h, key_head.size()[2], key_head.size()[3])
        value_head = self.transpose_for_scores1(value)

        scores = torch.matmul(query_head, key_head.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
            # scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value_head)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return x

class IGSAN(nn.Module):
    def __init__(self, num_layers, embed_size, h=1, is_share=False, drop=None):
        super(IGSAN, self).__init__()
        self.num_layers = num_layers
        self.bns = clones(nn.BatchNorm1d(embed_size), num_layers)
        self.dropout = clones(nn.Dropout(drop), num_layers)
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.att_layers = clones(IntrativeSelfattention(embed_size, h, is_share, drop=drop), num_layers)

        self.fc_in = nn.Linear(embed_size, embed_size)
        self.bn_in = nn.BatchNorm1d(embed_size)
        self.dropout_in = nn.Dropout(0.2)

        self.fc_int = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)
        self.bn_out = nn.BatchNorm1d(embed_size)
        self.dropout_out = nn.Dropout(0.2)

    def forward(self, q, k, v, mask=None):
        ''' imb_emb -- (bs, num_r, dim), pos_emb -- (bs, num_r, num_r, dim) '''
        bs, num_r, emb_dim = q.size()

        # 1st layer
        attention_output = self.att_layers[0](q, k, v, mask)  # (bs, r, d)
        attention_output = self.fc_in(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.bn_in((attention_output + q).permute(0, 2, 1)).permute(0, 2, 1)
        intermediate_output = self.fc_int(attention_output)
        intermediate_output = F.relu(intermediate_output)
        intermediate_output = self.fc_out(intermediate_output)
        intermediate_output = self.dropout_out(intermediate_output)
        graph_output = self.bn_out((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)
        # x = (self.bns[0](x.view(bs*num_r, -1))).view(bs, num_r, -1)
        # agsa_emb = q + self.dropout[0](x)

        # # 2nd~num_layers
        # for i in range(self.num_layers - 1):
        #     x = self.att_layers[i+1](agsa_emb, mask) #(bs, r, d)
        #     x = (self.bns[i+1](x.view(bs*num_r, -1))).view(bs, num_r, -1)
        #     agsa_emb = agsa_emb + self.dropout[i+1](x)

        return graph_output

# ### Glove encoderText
# class EncoderText(nn.Module):

#     def __init__(self, opt):
#         super(EncoderText, self).__init__()
#         self.embed_size = opt.embed_size
#         # word embedding
#         self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
#         # caption embedding
#         self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)
#         vocab = pickle.load(open('vocab/'+opt.data_name+'_vocab.pkl', 'rb'))
#         word2idx = vocab.word2idx
#         # self.init_weights()
#         self.init_weights('glove', word2idx, opt.word_dim)
#         self.dropout = nn.Dropout(0.1)

#     def init_weights(self, wemb_type, word2idx, word_dim):
#         if wemb_type.lower() == 'random_init':
#             nn.init.xavier_uniform_(self.embed.weight)
#         else:
#             # Load pretrained word embedding
#             if 'fasttext' == wemb_type.lower():
#                 wemb = torchtext.vocab.FastText()
#             elif 'glove' == wemb_type.lower():
#                 wemb = torchtext.vocab.GloVe()
#             else:
#                 raise Exception('Unknown word embedding type: {}'.format(wemb_type))
#             assert wemb.vectors.shape[1] == word_dim

#             # quick-and-dirty trick to improve word-hit rate
#             missing_words = []
#             for word, idx in word2idx.items():
#                 if word not in wemb.stoi:
#                     word = word.replace('-', '').replace('.', '').replace("'", '')
#                     if '/' in word:
#                         word = word.split('/')[0]
#                 if word in wemb.stoi:
#                     self.embed.weight.glo_data[idx] = wemb.vectors[wemb.stoi[word]]
#                 else:
#                     missing_words.append(word)
#             print('Words: {}/{} found in vocabulary; {} words missing'.format(
#                 len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

#     def forward(self, x, lengths):
#         # return out
#         x = self.embed(x) #->(128,29,300)
#         x = self.dropout(x)

#         packed = pack_padded_sequence(x, lengths, batch_first=True)

#         # Forward propagate RNN
#         out, _ = self.rnn(packed)

#         # Reshape *final* output to (batch_size, hidden_size)
#         padded = pad_packed_sequence(out, batch_first=True)
#         cap_emb, cap_len = padded

#         cap_emb = l2norm(cap_emb, dim=-1) #(128,29,1024)
#         cap_emb_mean = torch.mean(cap_emb, 1)
#         cap_emb_mean = l2norm(cap_emb_mean) #(128,1024)

#         return cap_emb


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions(batch_size, region_num, emb_size) and raw global image(batch_size, emb_size)
        l_emb = self.embedding_local(local)  # batch_norm for region_num
        g_emb = self.embedding_global(raw_global)  # batch_norm for emb_sieze

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)  # (batch_size, region_num, emb_size)
        common = l_emb.mul(g_emb)  # (batch_size, region_num, emb_size)
        weights = self.embedding_common(common).squeeze(2)  # (batch_size, region_num)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = weights.unsqueeze(2) * local
        new_global = new_global.sum(dim=1)
        # new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_relation(query, context, smooth=9., eps=1e-8):
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)
    query_norm = torch.norm(query, p=2, dim=2).repeat(1, queryL).view(batch_size_q, queryL, queryL).clamp(min=1e-8)
    source_norm = torch.norm(context, p=2, dim=2).repeat(1, sourceL).view(batch_size_q, sourceL, sourceL).clamp(
        min=1e-8)
    attn = torch.div(attn, query_norm)
    attn = torch.div(attn, source_norm)

    return attn


def get_TAposition(depend, lens):
    temlen = max(lens)
    adj = np.zeros((len(lens), temlen, temlen))
    for j in range(len(depend)):
        dep = depend[j]
        for i, pair in enumerate(dep):
            if i == 0 or pair[0] >= temlen or pair[1] >= temlen:
                continue
            adj[j, pair[0], pair[1]] = 1
            adj[j, pair[1], pair[0]] = 1
        adj[j] = adj[j] + np.eye(temlen)

    return torch.from_numpy(adj).cuda().float()

class Aggregation_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Aggregation_regulator, self).__init__()

        self.rar_q_w = nn.Sequential(nn.Linear(sim_dim, sim_dim),
                                     nn.Tanh(),
                                     nn.Dropout(0.4))
        self.rar_k_w = nn.Sequential(nn.Linear(sim_dim, sim_dim),
                                     nn.Tanh(),
                                     nn.Dropout(0.4))
        self.rar_v_w = nn.Sequential(nn.Linear(sim_dim, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, mid, hig):

        mid_k = self.rar_k_w(mid)
        hig_q = self.rar_q_w(hig)
        hig_q = hig_q.unsqueeze(1).repeat(1, mid_k.size(1), 1)

        weights = mid_k.mul(hig_q)
        weights = self.softmax(self.rar_v_w(weights).squeeze(2))

        new_hig = (weights.unsqueeze(2) * mid).sum(dim=1)
        new_hig = l2norm(new_hig, dim=-1)

        return new_hig

class Correpondence_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Correpondence_regulator, self).__init__()

        self.rcr_smooth_w = nn.Sequential(nn.Linear(sim_dim, sim_dim // 2),
                                          nn.Tanh(),
                                          nn.Linear(sim_dim // 2, 1))
        self.rcr_matrix_w = nn.Sequential(nn.Linear(sim_dim, sim_dim * 2),
                                          nn.Tanh(),
                                          nn.Linear(sim_dim * 2, embed_dim))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, matrix, smooth):
        matrix = matrix.to(x.device)

        matrix = (self.tanh(self.rcr_matrix_w(x)) + matrix).clamp(min=-1, max=1)
        smooth = self.relu(self.rcr_smooth_w(x) + smooth)

        return matrix, smooth

def cross_attention(query, context, matrix, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    device = query.device  # 获取 query 所在设备
    matrix = matrix.to(device)  # 将 matrix 移动到同一设备

    query = torch.mul(query, matrix)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, dim=-1)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=2)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    wcontext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    wcontext = torch.transpose(wcontext, 1, 2)
    wcontext = l2norm(wcontext, dim=-1)

    return wcontext


class Alignment_vector(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Alignment_vector, self).__init__()

        self.sim_transform_w = nn.Linear(embed_dim, sim_dim)

    def forward(self, query, context, matrix, smooth):

        wcontext = cross_attention(query, context, matrix, smooth)
        sim_rep = torch.pow(torch.sub(query, wcontext), 2)
        sim_rep = l2norm(self.sim_transform_w(sim_rep), dim=-1)

        return sim_rep


class ADAPT(nn.Module):

    def __init__(
            self, k=None, q1_size=None, q2_size=None, v1_size=None, v2_size=None,
            nonlinear_proj=False, groups=1, sg_dim=None,
    ):
        '''
            value_size (int): size of the features from the value matrix
            query_size (int): size of the global query vector
            k (int, optional): only used for non-linear projection
            nonlinear_proj (bool): whether to project gamma and beta non-linearly
            groups (int): number of feature groups (default=1)
        '''
        super().__init__()

        # self.query_size = query_size
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

            # Q1 adapter
            if q1_size != v1_size:
                self.v1_transform = nn.Sequential(
                    nn.Linear(v1_size, q1_size),
                )
                v1_size = q1_size

            self.fc_gamma = nn.Sequential(
                nn.Linear(q1_size, v1_size // groups),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(q1_size, v1_size // groups),
            )

            # V2 adapter
            # if v2_size is not None:

            #     self.imgsg_beta = nn.Sequential(
            #         nn.Linear(v2_size, v1_size//groups),
            #     )

            # Q2 adapter
            # if q2_size is not None:

            #     self.txtsg_beta = nn.Sequential(
            #         nn.Linear(q2_size, v1_size // groups),
            #     )

    def forward(self, value1, value2, query1, query2):

        # value 1 (img)
        B, D, rk = value1.shape
        Bv, Dv = query1.shape

        # print(value1.shape)
        # print(value2.shape)
        # print(D)
        # print(Dv)
        if D != Dv:
            value1 = value1.permute(0, 2, 1)
            value1 = self.v1_transform(value1).permute(0, 2, 1)  # B, Dv, K

        value1 = value1.view(
            B, Dv // self.groups, self.groups, -1
        )

        # value 2 (imgsg)
        # if value2 is not None:
        #     v2_betas = value2.view(
        #         B, Dv//self.groups, 1, 1
        #     )
        # query1 (caption)
        gammas = self.fc_gamma(query1).view(
            Bv, Dv // self.groups, 1, 1
        )
        betas = self.fc_beta(query1).view(
            Bv, Dv // self.groups, 1, 1
        )

        # query2 (txtsg)
        # if query2 is not None:
        #     q2_betas = self.txtsg_beta(query2).view(
        #         Bv, Dv//self.groups, 1, 1
        #     )

        if query2 is not None and value2 is None:  # sg_type: txt or bi_concat
            normalized = value1 * (gammas + 1) + betas
        elif query2 is None and value2 is not None:  # sg_type: img
            normalized = value1 * (gammas + 1) + betas
        else:  # sg_type: bi_adapt
            normalized = value1 * (gammas + 1) + betas

        normalized = normalized.view(B, Dv, -1)
        return normalized


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, opt, embed_size, sim_dim, module_name='AVE', sgr_step=3, focal_type='equal',  q1_size=1024, q2_size=1024, v1_size=1024, v2_size=1024, k=1):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name
        self.opt = opt
        self.embed_dim = embed_size
        self.sim_dim = sim_dim
        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        self.sim_trancon_w = nn.Linear(embed_size, sim_dim)

        if opt.self_regulator == 'only_rar':
            rar_step, rcr_step, alv_step = opt.rar_step, 0, 1
        elif opt.self_regulator == 'only_rcr':
            rar_step, rcr_step, alv_step = 0, opt.rcr_step, opt.rcr_step
        elif opt.self_regulator == 'coop_rcar':
            rar_step, rcr_step, alv_step = opt.rcar_step, opt.rcar_step-1, opt.rcar_step
        else:
            raise ValueError('Something wrong with opt.self_regulator')

        # 假设你从 opt 里能拿到 embed_size（就是 d_model）
        self.fusion_gate = nn.Linear(opt.embed_size * 2, opt.embed_size)

        self.rar_modules = nn.ModuleList([Aggregation_regulator(sim_dim, embed_size) for i in range(rar_step)])
        self.rcr_modules = nn.ModuleList([Correpondence_regulator(sim_dim, embed_size) for j in range(rcr_step)])
        self.alv_modules = nn.ModuleList([Alignment_vector(sim_dim, embed_size) for m in range(alv_step)])

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.focal_type = focal_type

        # self.glo_model = glo_model

        self.scan_attention = SCAN_attention(embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)

        self.cap_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        self.adapt_txt = ADAPT(k, v1_size, v2_size, q1_size, q2_size, nonlinear_proj=True)
        # self.cap_pool = weightpool(opt)

        self.init_weights()

    def forward(self, opt, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        # # img_glo  (64,1024)
        img_glo = self.v_global_w(img_emb, img_ave)


        for i in range(n_caption):

            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)  # (1, L, D_text)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)  # (n_img, L, D_text)

            # 文本引导信号
            cap_guide = cap_i.mean(dim=1)  # (1, D_text)
            cap_guide_expand = cap_guide.repeat(n_image, 1)  # (n_img, D_text)

            # 图像特征文本引导适配
            img_emb_adapted = self.adapt_txt(
                img_emb.permute(0, 2, 1),  # (n_img, num_boxes, img_dim)
                img_glo,  # 可选
                cap_guide_expand,  # (n_img, D_text)
                None
            )
            new_img_emb = img_emb_adapted.permute(0, 2, 1)

            query = cap_i_expand if opt.attn_type == 't2i' else new_img_emb
            context = new_img_emb if opt.attn_type == 't2i' else cap_i_expand

            smooth = self.opt.t2i_smooth if self.opt.attn_type == 't2i' else self.opt.i2t_smooth
            matrix = torch.ones(self.embed_dim)

            for m, rar_module in enumerate(self.rar_modules):
                sim_mid = self.alv_modules[m](query, context, matrix, smooth)
                if m == 0:
                    sim_hig = torch.mean(sim_mid, 1)
                if m < (self.opt.rcar_step - 1):
                    matrix, smooth = self.rcr_modules[m](sim_mid, matrix, smooth)
                sim_hig = rar_module(sim_mid, sim_hig)
            sim_i = self.sigmoid(self.sim_eval_w(sim_hig))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all


    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SCAN_attention(nn.Module):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """

    def __init__(self, embed_size):
        super(SCAN_attention, self).__init__()
        self.fcq = nn.Linear(embed_size, embed_size)
        self.fck = nn.Linear(embed_size, embed_size)
        self.fcv = nn.Linear(embed_size, embed_size)

    def forward(self, query, context, smooth, eps=1e-8):
        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        sim_k = self.fck(context)
        sim_q = self.fcq(query)
        sim_v = context
        attn = torch.bmm(sim_k, sim_q.permute(0, 2, 1))

        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = F.softmax(attn.permute(0, 2, 1) * smooth, dim=2)

        # --> (batch, queryL, d)
        weightedContext = torch.bmm(attn, sim_v)
        weightedContext = l2norm(weightedContext, dim=-1)

        return weightedContext




class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.CE = nn.CrossEntropyLoss()
        self.T = 0.05

    def forward(self, scores):
        batch_size = scores.size(0)
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        cost_s /= self.T
        cost_im /= self.T

        labels = torch.Tensor(list(range(batch_size))).long().cuda()

        return (self.CE(cost_im, labels) + self.CE(cost_s, labels)) / 2

class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2

class GAT_111(nn.Module):
    def __init__(self, opt):
        super(GAT_111, self).__init__()

        config_cap = GATopt(opt.embed_size, 1)
        config_cap1 = GATopt(opt.embed_size, 1)
        self.gat_cap = GAT1(config_cap)
        self.gat_img = GAT1(config_cap1)
        self.gat_cat_1 = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.gat_cat_2 = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.ln = nn.LayerNorm(opt.embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(opt.embed_size, opt.embed_size * 4),
            nn.ReLU(),
            nn.Linear(opt.embed_size * 4, opt.embed_size)
        )
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)


        self.ln1 = nn.LayerNorm(opt.embed_size)
        self.ln2 = nn.LayerNorm(opt.embed_size)
        self.dropout = nn.Dropout(dropout)

        # 残差加权
        self.alpha_img = nn.Parameter(torch.tensor(0.5))
        self.alpha_cap = nn.Parameter(torch.tensor(0.5))

        # 可选线性层对 GAT 输出进行映射
        self.img_linear = nn.Linear(opt.embed_size, opt.embed_size)
        self.cap_linear = nn.Linear(opt.embed_size, opt.embed_size)

        # LayerNorm
        self.img_ln = nn.LayerNorm(opt.embed_size)
        self.cap_ln = nn.LayerNorm(opt.embed_size)

    def forward(self, img_emb0, cap_emb0):

        img_gat= self.gat_img(img_emb0, img_emb0, img_emb0)  # multi-head attention
        img_gat = self.img_linear(img_gat)                        # 线性映射
        img_feat = img_emb0 + self.alpha_img * img_gat            # 加权残差
        img_feat = self.img_ln(self.dropout(img_feat))           # LayerNorm + Dropout
        img_emb = l2norm(img_feat, dim=-1)                       # L2归一化

        # -----------------------
        # 文本 GAT
        # -----------------------
        cap_gat = self.gat_cap(cap_emb0, cap_emb0, cap_emb0)
        cap_gat = self.cap_linear(cap_gat)
        cap_feat = cap_emb0 + self.alpha_cap * cap_gat
        cap_feat = self.cap_ln(self.dropout(cap_feat))
        cap_emb = l2norm(cap_feat, dim=-1)

        return img_emb, cap_emb

class GAT1(nn.Module):
    def __init__(self, config_gat):
        super(GAT1, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, querys, keys, values, attention_mask=None, position_weight = None):
        # hidden_states = querys
        # for layer_module in self.encoder:
        #
        #     hidden_states = layer_module(querys, keys, values, attention_mask, position_weight)
        hidden_states = querys
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, hidden_states, hidden_states, attention_mask, position_weight)

        return hidden_states  # B, seq_len, D


class CSAN(nn.Module):
    """
    Similarity Reasoning and Filtration (CSAN) Network
    """

    def __init__(self, opt):
        super(CSAN, self).__init__()
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt)
        self.sim_enc = EncoderSimilarity(opt, opt.embed_size, opt.sim_dim,
                                         opt.module_name, opt.sgr_step, opt.focal_type)
        self.GAT_model = GAT_111(opt)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            # cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        # params = list(self.txt_enc.parameters())
        # params += list(self.img_enc.parameters())
        # params += list(self.sim_enc.parameters())
        self.bert_params = list(self.txt_enc.bert.parameters())
        self.other_params = list(self.img_enc.parameters()) + list(self.txt_enc.fc.parameters()) + list(
            self.sim_enc.parameters())
        self.params = [self.bert_params, self.other_params]

        self.bert_lr = opt.bert_lr
        self.other_lr = opt.other_lr
        self.lr = [self.bert_lr, self.other_lr]

        # self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.bert_optimizer = torch.optim.Adam(self.bert_params, lr=self.bert_lr)
        self.other_optimizer = torch.optim.Adam(self.other_params, lr=self.other_lr)
        self.optimizer = [self.bert_optimizer, self.other_optimizer]

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict(), self.GAT_model.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])
        self.GAT_model.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()
        self.GAT_model.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()
        self.GAT_model.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        # cap_embs = self.txt_enc(captions)
        cap_embs = self.txt_enc(captions, lengths)
        img_emb, cap_emb = self.GAT_model(img_embs, cap_embs)

        # img_embs = self.GAT

        return img_emb, cap_emb, lengths

    def forward_sim(self, opt, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(opt, img_embs, cap_embs, cap_lens)
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, opt, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        # self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('bert_lr', self.optimizer[0].param_groups[0]['lr'])
        self.logger.update('other_lr', self.optimizer[1].param_groups[0]['lr'])
        # self.logger.update('bert_lr', self.lr[0])  #
        # self.logger.update('other_lr', self.lr[1])  #

        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(opt, img_embs, cap_embs, cap_lens)

        # self.optimizer.zero_grad()
        for optimizer in self.optimizer:
            optimizer.zero_grad()
        loss = self.forward_loss(sims)

        loss.backward()

        self.optimizer[0].step()

        self.optimizer[1].step()