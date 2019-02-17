import torch
import torch.nn as nn
from options import opt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as functional

class CharBiLSTM(nn.Module):
    def __init__(self, embedding):
        super(CharBiLSTM, self).__init__()

        self.hidden_dim = opt.char_hidden_dim
        if opt.bidirect:
            self.hidden_dim = self.hidden_dim // 2
        self.char_drop = nn.Dropout(opt.dropout)

        self.char_embeddings = nn.Embedding(embedding.size(0), embedding.size(1), padding_idx=0)
        self.char_embeddings.weight.data.copy_(embedding)

        self.char_lstm = nn.LSTM(embedding.size(1), self.hidden_dim, num_layers=1, batch_first=True, bidirectional=opt.bidirect)
        if opt.gpu >= 0 and torch.cuda.is_available():
            self.char_drop = self.char_drop.cuda(opt.gpu)
            self.char_embeddings = self.char_embeddings.cuda(opt.gpu)
            self.char_lstm = self.char_lstm.cuda(opt.gpu)


    def forward(self, input, seq_lengths):

        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)

        return char_hidden[0].transpose(1,0).contiguous().view(batch_size,-1)


class WordRep(nn.Module):
    def __init__(self, word_emb, pos_emb, char_emb):
        super(WordRep, self).__init__()

        if opt.use_char:
            self.char_feature = CharBiLSTM(char_emb)

        self.drop = nn.Dropout(opt.dropout)

        self.word_embedding = nn.Embedding(word_emb.size(0), word_emb.size(1), padding_idx=0)
        self.word_embedding.weight.data.copy_(word_emb)

        if pos_emb is not None:
            self.use_pos = True
            self.pos_embedding = nn.Embedding(pos_emb.size(0), pos_emb.size(1), padding_idx=0)
            self.pos_embedding.weight.data.copy_(pos_emb)
        else:
            self.use_pos = False

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.drop = self.drop.cuda(opt.gpu)
            self.word_embedding = self.word_embedding.cuda(opt.gpu)
            if pos_emb is not None:
                self.pos_embedding = self.pos_embedding.cuda(opt.gpu)


    def forward(self, word_inputs, pos_inputs, char_inputs, char_seq_lengths, char_seq_recover):

        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embedding(word_inputs)
        word_list = [word_embs]

        if self.use_pos:
            pos_embs = self.pos_embedding(pos_inputs)
            word_list.append(pos_embs)

        if opt.use_char:

            char_features = self.char_feature.forward(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_list.append(char_features)


        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent


class Encoder(nn.Module):
    def __init__(self, word_emb, pos_emb, char_emb):
        super(Encoder, self).__init__()

        self.droplstm = nn.Dropout(opt.dropout)
        self.wordrep = WordRep(word_emb, pos_emb, char_emb)

        self.input_size = word_emb.size(1)
        if pos_emb is not None:
            self.input_size += opt.pos_emb_dim
        if opt.use_char:
            self.input_size += opt.char_hidden_dim

        if opt.bidirect:
            lstm_hidden = opt.hidden_dim // 2
        else:
            lstm_hidden = opt.hidden_dim

        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=opt.bidirect)

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(opt.gpu)
            self.lstm = self.lstm.cuda(opt.gpu)


    def forward(self, word_inputs, pos_inputs, char_inputs, char_seq_lengths, char_seq_recover):

        word_represent = self.wordrep(word_inputs, pos_inputs, char_inputs, char_seq_lengths, char_seq_recover)

        hidden = None
        lstm_out, hidden = self.lstm(word_represent, hidden)

        outputs = self.droplstm(lstm_out)

        return outputs, hidden

    def free_emb(self):
        freeze_net(self.wordrep.word_embedding)
        if opt.use_char:
            freeze_net(self.wordrep.char_feature.char_embeddings)


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.W = self.W.cuda(opt.gpu)

    def forward(self, dec_inputs, encoder_outputs):
        batch_size, dec_seq_len, _ = dec_inputs.size()
        flat_dec_inputs = dec_inputs.contiguous().view(-1, self.input_size)
        logits = self.W(flat_dec_inputs).view(batch_size , dec_seq_len, -1)
        logits = logits.bmm(encoder_outputs.transpose(2, 1)) # batch, dec, enc

        alphas = functional.softmax(logits, dim=2)

        output = torch.bmm(alphas, encoder_outputs)
        return output


class Decoder(nn.Module):
    def __init__(self, word_emb, char_emb, label_alphabet):
        super(Decoder, self).__init__()

        self.droplstm = nn.Dropout(opt.dropout)
        self.wordrep = WordRep(word_emb, None, char_emb)

        self.input_size = word_emb.size(1)
        if opt.use_char:
            self.input_size += opt.char_hidden_dim

        self.attn = Attention(self.input_size, opt.hidden_dim)

        # decoder use single-direction RNN
        self.lstm = nn.LSTM(opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        self.hidden2tag = nn.Linear(opt.hidden_dim, label_alphabet.size())

        # don't compute pad as loss
        self.loss_function = nn.NLLLoss(ignore_index=0, size_average=False)

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(opt.gpu)
            self.lstm = self.lstm.cuda(opt.gpu)
            self.hidden2tag = self.hidden2tag.cuda(opt.gpu)

    def forward_teacher_forcing(self, encoder_outputs, encoder_hidden,
                                word_inputs, label_tensor,
                                char_inputs, char_seq_lengths, char_seq_recover):

        word_represent = self.wordrep(word_inputs, None, char_inputs, char_seq_lengths,
                                      char_seq_recover)

        word_represent = self.attn(word_represent, encoder_outputs)

        if opt.bidirect: # encoder is bidirect, decoder is single-direct
            encoder_hidden_0 = encoder_hidden[0].transpose(1, 0).view(1, 1, -1).transpose(1, 0)
            encoder_hidden_1 = encoder_hidden[1].transpose(1, 0).view(1, 1, -1).transpose(1, 0)
            encoder_hidden = (encoder_hidden_0, encoder_hidden_1)

        lstm_out, hidden = self.lstm(word_represent, encoder_hidden)
        outputs = self.droplstm(lstm_out)

        outputs = self.hidden2tag(outputs)

        batch_size, dec_seq_len = word_inputs.size()

        outs = outputs.view(batch_size * dec_seq_len, -1)
        score = functional.log_softmax(outs, 1)
        total_loss = self.loss_function(score, label_tensor.view(batch_size * dec_seq_len))
        total_loss = total_loss / batch_size

        return total_loss, score

    def free_emb(self):
        freeze_net(self.wordrep.word_embedding)
        if opt.use_char:
            freeze_net(self.wordrep.char_feature.char_embeddings)



def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False
