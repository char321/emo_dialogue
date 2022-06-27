import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, g_hist, utter=None):
        """
        g_hist: (seq_len, batch, vector)
        utter: dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(g_hist)  # seq_len * batch * vector -> seq_len * batch * 1
        att = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch * 1 * seq_len
        context = torch.bmm(att, g_hist.transpose(0, 1))[:, 0, :]  # batch * (1 * seq_len) X batch * (seq_len * vector) -> batch * vector

        return context, att


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        """
        mem_dim: d_g - dim of global state
        cand_dim: d_m - dim of utterance representation
        alpha_dim: d_a ?
        """
        super(MatchingAttention, self).__init__()

        assert att_type != 'concat' or alpha_dim != None  # when att_type == 'concat', alpha_dim must != None
        assert att_type != 'dot' or mem_dim == cand_dim  # when att_type == 'dot', mem_dim must == cand_dim

        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type

        if att_type == 'general':
            # no bias
            self.transform_no_bias = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            # with bias
            self.transform_with_bias = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform_concat = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.linear = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, g_hist, utter, mask=None):
        """
        attention score over previous global state
        g_hist: history of global state -> (seq_len, batch, mem_dim), seq_len is time step t
        utter: utterance -> (batch, cand_dim)
        mask: mask of utterances which indicate effective part (length) -> (batch, seq_len)
        """

        if type(None) == type(mask):
            mask = torch.ones(g_hist.size(1), g_hist.size(0)).type(g_hist.type())

        if self.att_type == 'dot':
            # dot attention
            # d_g == d_m
            temp_g = g_hist.permute(1, 2, 0)  # batch * d_g * seq_len
            temp_utter = utter.unsqueeze(1)  # batch * 1 * d_m
            att = F.softmax(torch.bmm(temp_utter, temp_g), dim=2)  # batch * 1 * seq_len
        elif self.att_type == 'general':
            # general attention
            temp_g = g_hist.permute(1, 2, 0)  # batch * d_g * seq_len
            temp_utter = self.transform_no_bias(utter).unsqueeze(1)  # batch * 1 * d_g
            att = F.softmax(torch.bmm(temp_utter, temp_g), dim=2)  # batch * 1 * seq_len

        elif self.att_type == 'general2':
            # masked attention
            temp_g = g_hist.permute(1, 2, 0)  # batch * d_g * seq_len
            temp_utter = self.transform_with_bias(utter).unsqueeze(1)  # batch * 1 * d_g
            alpha_ = F.softmax((torch.bmm(temp_utter, temp_g)) * mask.unsqueeze(1), dim=2)  # batch * 1 * seq_len
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch * 1 * seq_len
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch * 1 * 1
            att = alpha_masked / alpha_sum  # normalization -> batch * 1 * seq_len
            # import ipdb;ipdb.set_trace()
        else:
            # concatenation
            temp_g = g_hist.transpose(0, 1)  # batch * seq_len * d_g
            temp_utter = utter.unsqueeze(1).expand(-1, g_hist.size()[0], -1)  # expand the dim: batch * seq_len * d_m
            cat_g_utter = torch.cat([temp_g, temp_utter], 2)  # batch * seq_len * (d_g + d_m)
            temp_res = F.tanh(self.transform_concat(cat_g_utter))  # batch * seq_len * alpha_dim
            att = F.softmax(self.linear(temp_res), 1).transpose(1, 2)  # batch * 1 * seq_len

        context = torch.bmm(att, g_hist.transpose(0, 1))[:, 0, :]  # batch * d_g

        return context, att


class DialogueRNNCell(nn.Module):

    def __init__(self, d_m, d_g, d_p, d_e, listener_state=False,
                 context_attention='simple', d_a=100, dropout_rate=0.5):
        super(DialogueRNNCell, self).__init__()

        self.d_m = d_m  # dimension of the utterance representation - 100
        self.d_g = d_g  # dimension of global state - 500
        self.d_p = d_p  # dimension of party state (speaker update is sufficient) - 500
        self.d_e = d_e  # dimension of emotion representation - 300

        self.listener_state = listener_state
        # global state - capture context of a give utterance by jointly encoding utterance & speaker state
        self.g_cell = nn.GRUCell(d_m + d_p, d_g)
        # party state (speaker update) - track state of speakers by encoding current utterance & context from global state
        self.p_cell = nn.GRUCell(d_m + d_g, d_p)
        # emotion representation - infer from the speaker state
        self.e_cell = nn.GRUCell(d_p, d_e)
        # party state (listener update), which is unnecessary
        if listener_state:
            self.l_cell = nn.GRUCell(d_m + d_p, d_p)

        self.dropout = nn.Dropout(dropout_rate)

        # attention score
        if context_attention == 'simple':
            self.attention = SimpleAttention(d_g)
        else:
            self.attention = MatchingAttention(d_g, d_m, d_a, context_attention)

    def _select_parties(self, q, indices):
        """
        q: party state
        indices: max indices of party state mask for each batch
        """
        q_sel = []
        for idx, j in zip(indices, q):
            q_sel.append(j[idx].unsqueeze(0))  # get party state whose number is idx
        q_sel = torch.cat(q_sel, 0)  # concatenates the sequence of tensors
        return q_sel

    def forward(self, utter, q_mask, g_hist, q, e):
        """
        utter: utterances -> (batch, d_m)
        q_mask: party state mask -> (batch, party) - party = 2
        g_hist: previous global state -> (t-1, batch, d_g), whose t will increase at each training step
        q: party state -> (batch, party, d_p)
        e: emotion representation -> (batch, d_e)
        """
        num_party = q_mask.size()[1]
        num_batch = q_mask.size()[0]
        qm_idx = torch.argmax(q_mask, 1)  # max idx of qmax (for each batch), which is batch
        q0_sel = self._select_parties(q, qm_idx)  # batch * d_p

        if g_hist.size()[0] == 0:
            # first cell in GRU
            # initialize context vector
            context = torch.zeros(num_batch, self.d_g).type(utter.type())
            att = None
            hidden = torch.zeros(num_batch, self.d_g).type(utter.type())
        else:
            # get context vector using global state & current utterance through attention mechanism
            context, att = self.attention(g_hist, utter)  # batch * d_g, batch * 1 * seq_len
            hidden = g_hist[-1]

        # Global state
        '''
        input for g_cell:
        - input: concatenation of utterance and party state -> batch * (d_m + d_p)
        - hidden: hidden state -> batch * d_g 
        '''
        temp_g = self.g_cell(torch.cat([utter, q0_sel], dim=1), hidden)
        # torch.zeros(utter.size()[0], self.D_g).type(utter.type()) if g_hist.size()[0] == 0 else
        # g_hist[-1])
        res_g = self.dropout(temp_g)  # batch * d_g

        # Party state (speaker)
        temp_utter_context = torch.cat([utter, context], dim=1).unsqueeze(1).expand(-1, num_party, -1)  # batch * party * (d_m + d_g)
        '''
        input for p_cell:
        - input: concatenation of utterance and context vector -> (batch * party) * (d_m + d_g)
        - hidden: hidden state -> (batch * party) * d_p
        '''
        temp_speaker = self.p_cell(temp_utter_context.contiguous().view(-1, self.d_m + self.d_g), q.view(-1, self.d_p))
        temp_speaker = temp_speaker.view(utter.size()[0], -1, self.d_p)  # batch * party * d_p
        temp_speaker = self.dropout(temp_speaker)

        # Party state (listener)
        if self.listener_state:
            temp_utter = utter.unsqueeze(1).expand(-1, num_party, -1).contiguous().view(-1, self.d_m)  # (batch * party) * d_m
            # select party state of the corresponding party (defined by q_mask)
            temp_party_state = self._select_parties(temp_speaker, qm_idx).unsqueeze(1).expand(-1, num_party, -1).contiguous().view(-1, self.d_p)  # (batch * party) * d_p
            temp_utter_party_state = torch.cat([temp_utter, temp_party_state], 1)

            '''
            input for l_cell:
            - input: concatenation of utterance and context vector -> (batch * party) * (d_m + d_p)
            - hidden: hidden state -> (batch * party) * d_p
            '''
            temp_listener = self.l_cell(temp_utter_party_state, q.view(-1, self.d_p)).view(num_batch, -1, self.d_p)  # batch * party * d_p
            temp_listener = self.dropout(temp_listener)
        else:
            temp_listener = q
        temp_q_mask = q_mask.unsqueeze(2)  # batch * party * 1
        # Party state (updated)
        res_q = temp_listener * (1 - temp_q_mask) + temp_speaker * temp_q_mask  # batch * party * d_p

        # Emotion Representation
        e = torch.zeros(num_batch, self.d_e).type(utter.type()) if e.size()[0] == 0 else e
        '''
        input for e_cell:
        - input: (updated) party state of the corresponding party (defined by q_mask) -> batch * d_p
        - hidden: hidden state -> batch * d_e
        '''
        temp_e = self.e_cell(self._select_parties(res_q, qm_idx), e)
        res_e = self.dropout(temp_e)  # batch * d_e

        return res_g, res_q, res_e, att


class DialogueRNN(nn.Module):

    def __init__(self, d_m, d_g, d_p, d_e, listener_state=False,
                 context_attention='simple', d_a=100, dropout_rate=0.5):
        super(DialogueRNN, self).__init__()

        self.d_m = d_m
        self.d_g = d_g
        self.d_p = d_p
        self.d_e = d_e
        self.dropout = nn.Dropout(dropout_rate)

        self.dialogue_cell = DialogueRNNCell(d_m, d_g, d_p, d_e, listener_state, context_attention, d_a, dropout_rate)

    def forward(self, utters, q_masks):
        """
        utters: sequence of utterance -> (seq_len * batch * d_m)
        q_masks: party state mask -> (seq_len * batch * party)
        """

        # Initialization
        num_batch = q_masks.size()[1]
        num_party = q_masks.size()[2]
        g_hist = torch.zeros(0).type(utters.type())  # 0-dimensional tensor
        q = torch.zeros(num_batch, num_party, self.d_p).type(utters.type())  # (batch * party * d_p)
        temp_e = torch.zeros(0).type(utters.type())  # (batch * d_e)
        e = temp_e

        # list of history attention score
        att_list = []

        # iter all utterances
        for utter, q_mask in zip(utters, q_masks):
            g, q, temp_e, att = self.dialogue_cell(utter, q_mask, g_hist, q, temp_e)

            # append history global state
            g_hist = torch.cat([g_hist, g.unsqueeze(0)], 0)
            # append history emotion representation
            e = torch.cat([e, temp_e.unsqueeze(0)], 0)

            if type(att) != type(None):
                # att -> batch * 1 * seq_len, where seq_len will increase as time
                att_list.append(att[:, 0, :])  # append current attention score to the list

        return e, att_list  # seq_len * batch * d_e, list of (batch * seq_len)


class BiModel(nn.Module):

    def __init__(self, d_m, d_g, d_p, d_e, d_h,
                 n_classes=7, listener_state=False, context_attention='simple', d_a=100, rec_dropout_rate=0.5,
                 dropout_rate=0.5):
        super(BiModel, self).__init__()

        self.d_m = d_m
        self.d_g = d_g
        self.d_p = d_p
        self.d_e = d_e
        self.d_h = d_h
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout_rate)
        self.rec_dropout = nn.Dropout(dropout_rate + 0.15)  # use rec_dropout_rate ?
        self.dialog_rnn_f = DialogueRNN(d_m, d_g, d_p, d_e, listener_state, context_attention, d_a, rec_dropout_rate)
        self.dialog_rnn_b = DialogueRNN(d_m, d_g, d_p, d_e, listener_state, context_attention, d_a, rec_dropout_rate)
        self.linear = nn.Linear(2 * d_e, 2 * d_h)
        self.softmax_fc = nn.Linear(2 * d_h, n_classes)
        self.match_att = MatchingAttention(2 * d_e, 2 * d_e, att_type='general2')

    def _reverse_seq(self, utters, u_mask):
        """
        utters: sequence of utterances -> (seq_len * batch * dim)
        u_mask: mask of utterance, which indicate the effective length of utterance -> (batch * seq_len)
        """
        # reshape
        temp_utters = utters.transpose(0, 1)
        # sum of mask for each batch (get effective length of utterances)
        mask_sum = torch.sum(u_mask, 1).int()

        rev_utters = []
        for utter, l in zip(temp_utters, mask_sum):
            rev_utter = torch.flip(utter[:l], [0])
            rev_utters.append(rev_utter)

        return pad_sequence(rev_utters)

    def forward(self, utters, q_masks, u_mask, use_att=True):
        """
        utters: sequence of utterance -> (seq_len * batch * D_m)
        q_masks: mask of global state -> (seq_len * batch * party)
        u_mask: mask of utterance, which indicate the effective length of utterance -> (batch * seq_len)
        use_att: whether to use attention mechanism
        """

        # forward cell
        emotions_f, alpha_f = self.dialog_rnn_f(utters, q_masks)  # seq_len * batch * d_e, list of history attention score
        emotions_f = self.rec_dropout(emotions_f)

        # reverse seq of utterances
        rev_utters = self._reverse_seq(utters, u_mask)
        # reverse seq of q_masks
        rev_q_masks = self._reverse_seq(q_masks, u_mask)

        # backward cell
        emotions_b, alpha_b = self.dialog_rnn_b(rev_utters, rev_q_masks)  # seq_len * batch * d_e, list of history attention score
        emotions_b = self._reverse_seq(emotions_b, u_mask)
        emotions_b = self.rec_dropout(emotions_b)

        # concatenate forward and backward results
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)

        if use_att:
            # for each emotion representation e, attention is applied over all the emotion representations
            emotion_list = []
            att_list = []
            for e in emotions:
                # get emotion representation (context), and attention
                emotion, att = self.match_att(emotions, e, mask=u_mask)
                emotion_list.append(emotion.unsqueeze(0))
                # att -> batch * 1 * seq_len, where seq_len is fixed in this loop
                att_list.append(att[:, 0, :])
            emotion_list = torch.cat(emotion_list, dim=0)
            hidden = F.relu(self.linear(emotion_list))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.softmax_fc(hidden), 2)  # seq_len, batch, n_classes

        # print(len(att_list))
        # print(alpha_f[0].shape)

        if use_att:
            # att_list contains attention of emotion representation over all previous emotions
            # => it contains attention score with **same** shape, as seq_len is fixed in this step
            # alpha_f, alpha_b contains attention of utterance over global history during learning
            # => it contains attention score with **different** shape, as seq_len is increasing as time
            return log_prob, att_list, alpha_f, alpha_b
        else:
            return log_prob, [], alpha_f, alpha_b


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, u_mask):
        """
        pred: prediction -> (batch * seq_len) * n_classes
        target: target class-> (batch * seq_len)
        u_mask: mask that indicate effective length of utterance -> (batch * seq_len)
        """

        temp_mask = u_mask.view(-1, 1)  # (batch * seq_len) * 1
        if type(self.weight) == type(None):
            # unweighted
            loss = self.loss(pred * temp_mask, target) / torch.sum(u_mask)
        else:
            # weighted
            loss = self.loss(pred * temp_mask, target) / torch.sum(self.weight[target] * temp_mask.squeeze())

        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class CNNFeatureExtractor(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False

    def forward(self, x, umask):

        if torch.cuda.is_available():
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
            ByteTensor = torch.cuda.ByteTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor
            ByteTensor = torch.ByteTensor

        num_utt, batch, num_words = x.size()

        x = x.type(LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words)  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x)  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
        emb = emb.transpose(-2,
                            -1).contiguous()  # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)

        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1)  # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).type(FloatTensor)  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim)  # (num_utt, batch, 1) -> (num_utt, batch, 100)
        features = (features * mask)  # (num_utt, batch, 100) -> (num_utt, batch, 100)

        return features


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss


class BiE2EModel(nn.Module):

    def __init__(self, D_emb, D_m, D_g, D_p, D_e, D_h, word_embeddings,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(BiE2EModel, self).__init__()

        self.D_emb = D_emb
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        # self.dropout_rec = nn.Dropout(0.2)
        self.dropout_rec = nn.Dropout(dropout)
        self.turn_rnn = nn.GRU(D_emb, D_m)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.linear1 = nn.Linear(2 * D_e, D_h)
        # self.linear2     = nn.Linear(D_h, D_h)
        # self.linear3     = nn.Linear(D_h, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        self.embedding = nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1])
        self.embedding.weight.data.copy_(word_embeddings)
        self.embedding.weight.requires_grad = True
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, data, att2=False):

        # T1 = word_embeddings[data.turn1] # seq_len, batch, D_emb
        # T2 = word_embeddings[data.turn2] # seq_len, batch, D_emb
        # T3 = word_embeddings[data.turn3] # seq_len, batch, D_emb

        T1 = (self.embedding(data.turn1))
        T2 = (self.embedding(data.turn2))
        T3 = (self.embedding(data.turn3))

        T1_, h_out1 = self.turn_rnn(T1,
                                    torch.zeros(1, T1.size(1), self.D_m).type(T1.type()))
        T2_, h_out2 = self.turn_rnn(T2,
                                    torch.zeros(1, T1.size(1), self.D_m).type(T1.type()))
        T3_, h_out3 = self.turn_rnn(T3,
                                    torch.zeros(1, T1.size(1), self.D_m).type(T1.type()))

        U = torch.cat([h_out1, h_out2, h_out3], 0)  # 3, batch, D_m

        qmask = torch.FloatTensor([[1, 0], [0, 1], [1, 0]]).type(T1.type())
        qmask = qmask.unsqueeze(1).expand(-1, T1.size(1), -1)

        umask = torch.FloatTensor([[1, 1, 1]]).type(T1.type())
        umask = umask.expand(T1.size(1), -1)

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)  # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        # emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        # print(emotions)
        emotions = self.dropout_rec(emotions)

        # emotions = emotions.unsqueeze(1)
        if att2:
            att_emotion, _ = self.matchatt(emotions, emotions[-1])
            hidden = F.relu(self.linear1(att_emotion))
        else:
            hidden = F.relu(self.linear1(emotions[-1]))
        # hidden = F.relu(self.linear2(hidden))
        # hidden = F.relu(self.linear3(hidden))
        # hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), -1)  # batch, n_classes
        return log_prob


class E2EModel(nn.Module):

    def __init__(self, D_emb, D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(E2EModel, self).__init__()

        self.D_emb = D_emb
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        # self.dropout_rec = nn.Dropout(0.2)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.turn_rnn = nn.GRU(D_emb, D_m)
        self.dialog_rnn = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                      context_attention, D_a, dropout_rec)
        self.linear1 = nn.Linear(D_e, D_h)
        # self.linear2     = nn.Linear(D_h, D_h)
        # self.linear3     = nn.Linear(D_h, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

        self.matchatt = MatchingAttention(D_e, D_e, att_type='general2')

    def forward(self, data, word_embeddings, att2=False):

        T1 = word_embeddings[data.turn1]  # seq_len, batch, D_emb
        T2 = word_embeddings[data.turn2]  # seq_len, batch, D_emb
        T3 = word_embeddings[data.turn3]  # seq_len, batch, D_emb

        T1_, h_out1 = self.turn_rnn(T1,
                                    torch.zeros(1, T1.size(1), self.D_m).type(T1.type()))
        T2_, h_out2 = self.turn_rnn(T2,
                                    torch.zeros(1, T1.size(1), self.D_m).type(T1.type()))
        T3_, h_out3 = self.turn_rnn(T3,
                                    torch.zeros(1, T1.size(1), self.D_m).type(T1.type()))

        U = torch.cat([h_out1, h_out2, h_out3], 0)  # 3, batch, D_m

        qmask = torch.FloatTensor([[1, 0], [0, 1], [1, 0]]).type(T1.type())
        qmask = qmask.unsqueeze(1).expand(-1, T1.size(1), -1)

        emotions, _ = self.dialog_rnn(U, qmask)  # seq_len, batch, D_e
        # print(emotions)
        emotions = self.dropout_rec(emotions)

        # emotions = emotions.unsqueeze(1)
        if att2:
            att_emotion, _ = self.matchatt(emotions, emotions[-1])
            hidden = F.relu(self.linear1(att_emotion))
        else:
            hidden = F.relu(self.linear1(emotions[-1]))
        # hidden = F.relu(self.linear2(hidden))
        # hidden = F.relu(self.linear3(hidden))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), -1)  # batch, n_classes
        return log_prob


class Model(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(Model, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        # self.dropout_rec = nn.Dropout(0.2)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dialog_rnn = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                      context_attention, D_a, dropout_rec)
        self.linear1 = nn.Linear(D_e, D_h)
        # self.linear2     = nn.Linear(D_h, D_h)
        # self.linear3     = nn.Linear(D_h, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

        self.matchatt = MatchingAttention(D_e, D_e, att_type='general2')

    def forward(self, U, qmask, umask=None, att2=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions = self.dialog_rnn(U, qmask)  # seq_len, batch, D_e
        # print(emotions)
        emotions = self.dropout_rec(emotions)

        # emotions = emotions.unsqueeze(1)
        if att2:
            att_emotions = []
            for t in emotions:
                att_emotions.append(self.matchatt(emotions, t, mask=umask)[0].unsqueeze(0))
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear1(att_emotions))
        else:
            hidden = F.relu(self.linear1(emotions))
        # hidden = F.relu(self.linear2(hidden))
        # hidden = F.relu(self.linear3(hidden))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)  # seq_len, batch, n_classes
        return log_prob


class AVECModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h, attr, listener_state=False,
                 context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(AVECModel, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.attr = attr
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        self.dialog_rnn = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                      context_attention, D_a, dropout_rec)
        self.linear = nn.Linear(D_e, D_h)
        self.smax_fc = nn.Linear(D_h, 1)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions, _ = self.dialog_rnn(U, qmask)  # seq_len, batch, D_e
        emotions = self.dropout_rec(emotions)
        hidden = torch.tanh(self.linear(emotions))
        hidden = self.dropout(hidden)
        if self.attr != 4:
            pred = (self.smax_fc(hidden).squeeze())  # seq_len, batch
        else:
            pred = (self.smax_fc(hidden).squeeze())  # seq_len, batch
        return pred.transpose(0, 1).contiguous().view(-1)


class DailyDialogueModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h,
                 vocab_size, n_classes=7, embedding_dim=300,
                 cnn_output_size=100, cnn_filters=50, cnn_kernel_sizes=(3, 4, 5), cnn_dropout=0.5,
                 listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5, att2=True):

        super(DailyDialogueModel, self).__init__()

        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, cnn_filters,
                                                      cnn_kernel_sizes, cnn_dropout)

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout_rec)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.linear = nn.Linear(2 * D_e, 2 * D_h)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')

        self.n_classes = n_classes
        self.smax_fc = nn.Linear(2 * D_h, n_classes)
        self.att2 = att2

    def init_pretrained_embeddings(self, pretrained_word_vectors):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, input_seq, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        U = self.cnn_feat_extractor(input_seq, umask)

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)  # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        if self.att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)  # seq_len, batch, n_classes
        return log_prob, alpha, alpha_f, alpha_b
