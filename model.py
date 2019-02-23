import torch
import torch.nn as nn
from options import opt, config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as functional
from dataload import generateDecoderInput
import logging

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

    def forward_one_step(self, input):
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        char_rnn_out, char_hidden = self.char_lstm(char_embeds, char_hidden)
        return char_hidden[0].transpose(1,0).contiguous().view(1, 1, -1)


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

    # for batch 1
    def forward(self, word_inputs, pos_inputs, char_inputs, char_seq_lengths, char_seq_recover):

        word_represent = self.wordrep(word_inputs, pos_inputs, char_inputs, char_seq_lengths, char_seq_recover)

        hidden = None
        lstm_out, hidden = self.lstm(word_represent, hidden)

        outputs = self.droplstm(lstm_out)

        return outputs, hidden

    def forward_batch(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):

        word_represent = self.wordrep(word_inputs, None, char_inputs, char_seq_lengths,
                                      char_seq_recover)

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        outputs = self.droplstm(lstm_out.transpose(1, 0))

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

        if opt.use_char:
            self.char_feature = CharBiLSTM(char_emb)

        self.drop_wordrep = nn.Dropout(opt.dropout)

        self.word_embedding = nn.Embedding(word_emb.size(0), word_emb.size(1), padding_idx=0)
        self.word_embedding.weight.data.copy_(word_emb)

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.drop_wordrep = self.drop_wordrep.cuda(opt.gpu)
            self.word_embedding = self.word_embedding.cuda(opt.gpu)


        self.input_size = word_emb.size(1)
        if opt.use_char:
            self.input_size += opt.char_hidden_dim

        self.attn = Attention(self.input_size, opt.hidden_dim)

        # decoder use single-direction RNN
        self.lstm = nn.LSTM(opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.droplstm = nn.Dropout(opt.dropout)

        self.hidden2tag = nn.Linear(opt.hidden_dim, label_alphabet.size())

        # don't compute pad as loss
        self.loss_function = nn.NLLLoss(ignore_index=0, size_average=False)

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(opt.gpu)
            self.lstm = self.lstm.cuda(opt.gpu)
            self.hidden2tag = self.hidden2tag.cuda(opt.gpu)

    def forward_one_step(self, encoder_outputs, last_hidden, word_input, char_input):
        word_embs = self.word_embedding(word_input)
        word_list = [word_embs]

        if opt.use_char:
            char_features = self.char_feature.forward_one_step(char_input)
            word_list.append(char_features)

        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop_wordrep(word_embs)

        word_represent = self.attn(word_represent, encoder_outputs)

        lstm_out, hidden = self.lstm(word_represent, last_hidden)

        return lstm_out, hidden

    def forward_train(self, encoder_outputs, encoder_hidden,
                                word_inputs, label_tensor, char_inputs):

        if opt.bidirect: # encoder is bidirect, decoder is single-direct
            encoder_hidden_0 = encoder_hidden[0].transpose(1, 0).view(1, 1, -1).transpose(1, 0)
            encoder_hidden_1 = encoder_hidden[1].transpose(1, 0).view(1, 1, -1).transpose(1, 0)
            encoder_hidden = (encoder_hidden_0, encoder_hidden_1)

        last_hidden = encoder_hidden
        loss = 0
        correct = 0
        total = len(label_tensor)

        for di, label in enumerate(label_tensor):
            if opt.use_teacher_forcing:
                word_input = word_inputs[di]
                if opt.use_char:
                    char_input = char_inputs[di]
                else:
                    char_input = None
            else:
                raise RuntimeError("only support teacher-forcing training")

            lstm_out, hidden = self.forward_one_step(encoder_outputs, last_hidden, word_input, char_input)

            last_hidden = hidden

            output = self.droplstm(lstm_out)

            output = self.hidden2tag(output)

            score = functional.log_softmax(output.view(1, -1), dim=1)
            loss += self.loss_function(score, label)

            _, pred = torch.max(score, 1)
            correct += (pred == label).sum().item()


        return loss, total, correct


    def forward_infer(self, encoder_outputs, encoder_hidden, dec_word_alphabet, dec_char_alphabet):

        if opt.bidirect: # encoder is bidirect, decoder is single-direct
            encoder_hidden_0 = encoder_hidden[0].transpose(1, 0).view(1, 1, -1).transpose(1, 0)
            encoder_hidden_1 = encoder_hidden[1].transpose(1, 0).view(1, 1, -1).transpose(1, 0)
            encoder_hidden = (encoder_hidden_0, encoder_hidden_1)

        last_hidden = encoder_hidden

        preds = []
        word_string = '<SOS>'
        word_input, char_input = generateDecoderInput(word_string, dec_word_alphabet, dec_char_alphabet)

        while len(preds) <= int(config['dict_max_name_length']):

            lstm_out, hidden = self.forward_one_step(encoder_outputs, last_hidden, word_input, char_input)

            last_hidden = hidden

            output = self.droplstm(lstm_out)

            output = self.hidden2tag(output)

            score = functional.log_softmax(output.view(1, -1), dim=1)

            _, pred = torch.max(score, 1)

            pred = pred.item()

            if pred == 0 or pred == 1:  # we ignored pad and unk
                logging.debug("pred == 0 or pred == 1")
                continue

            token = dec_word_alphabet.get_instance(pred)
            if token == '<EOS>':
                break

            preds.append(token)

            word_input, char_input = generateDecoderInput(token, dec_word_alphabet, dec_char_alphabet)


        return preds


    def free_emb(self):
        freeze_net(self.word_embedding)
        if opt.use_char:
            freeze_net(self.char_feature.char_embeddings)



def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, lengths):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if opt.gpu >= 0 and torch.cuda.is_available():
            idxes = idxes.cuda(opt.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output

class AttnNet(nn.Module):
    def __init__(self, dictionary):
        super(AttnNet, self).__init__()

        self.attn = DotAttentionLayer(opt.hidden_dim)
        self.linear = nn.Linear(opt.hidden_dim, dictionary.id_alphabet.size())
        self.criterion = nn.CrossEntropyLoss()

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.attn = self.attn.cuda(opt.gpu)
            self.linear = self.linear.cuda(opt.gpu)

    def forward(self, encoder_outputs, lengths):
        output = self.attn(encoder_outputs, lengths)
        output = self.linear(output)

        return output

    def forward_train(self, encoder_outputs, lengths, label):
        score = self.forward(encoder_outputs, lengths)
        loss = self.criterion(score, label)

        total = label.size(0)
        _, pred = torch.max(score, 1)
        correct = (pred == label).sum().item()

        return loss, total, correct

    def forward_infer(self, encoder_outputs, lengths):
        score = self.forward(encoder_outputs, lengths)
        _, pred = torch.max(score, 1)
        return pred

    def free_emb(self):
        pass
