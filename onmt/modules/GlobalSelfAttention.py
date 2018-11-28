import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.Utils import aeq, sequence_mask


class GlobalSelfAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(self, dim, coverage=False, attn_type="dot", attn_hidden=0):
        super(GlobalSelfAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.attn_hidden = attn_hidden
        assert (self.attn_type in ["dot", "general", "mlp", "fine"]), (
                "Please select a valid attention type.")
        if attn_hidden > 0:
            self.transform_in = nn.Sequential(nn.Linear(dim, attn_hidden), nn.ELU(0.1))

        if self.attn_type == "general":
            d = attn_hidden if attn_hidden > 0 else dim
            self.linear_in = nn.Linear(d, d, bias=False)
        elif self.attn_type == "mlp":
            d = attn_hidden if attn_hidden > 0 else dim
            self.linear_context = nn.Linear(dim, d, bias=False)
            self.linear_query = nn.Linear(dim, d, bias=True)
            self.v = nn.Linear(d, 1, bias=False)
        elif self.attn_type == "fine":
            d = attn_hidden if attn_hidden > 0 else dim
            self.linear_context = nn.Linear(dim, d, bias=False)
            self.linear_query = nn.Linear(dim, d, bias=True)
            self.v = nn.Linear(d, dim, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type in ("mlp", "fine")
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.3)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_hidden > 0:
                h_t = self.transform_in(h_t)
                h_s = self.transform_in(h_s)

            if self.attn_type == "general":
                h_t = self.linear_in(h_t)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            d = self.attn_hidden if self.attn_hidden > 0 else dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, d)
            wq = wq.expand(tgt_batch, tgt_len, src_len, d)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, d)
            uh = uh.expand(src_batch, tgt_len, src_len, d)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            if self.attn_type == "mlp":
                return self.v(wquh.view(-1, d)).view(tgt_batch, tgt_len, src_len)
            elif self.attn_type == "fine":
                return self.v(wquh.view(-1, d)).view(tgt_batch, tgt_len, src_len, dim)

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(input, memory_bank)
        assert memory_lengths is not None
        mask = sequence_mask(memory_lengths)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        # mask the time step of self
        mask = mask.repeat(1, sourceL, 1)
        mask_self_index = list(range(sourceL))
        mask[:, mask_self_index, mask_self_index] = 0

        if self.attn_type == "fine":
            mask = mask.unsqueeze(3)

        align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        if self.attn_type == "fine":
            c = memory_bank.unsqueeze(1).mul(
                align_vectors).sum(dim=2, keepdim=False)
        else:
            c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, input], 2)
        attn_h = self.linear_out(concat_c)
        if self.attn_type in ["general", "dot"]:
            # attn_h = F.elu(attn_h, 0.1)
            # attn_h = F.elu(self.dropout(attn_h) + input, 0.1)

            # content selection gate
            attn_h = F.sigmoid(attn_h).mul(input)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return attn_h, align_vectors
