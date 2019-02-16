
from torch.utils.data import DataLoader, Dataset
import torch
from options import opt
import numpy as np

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def my_collate(input_batch_list):
    with torch.no_grad():
        batch_size = len(input_batch_list)
        enc_word = [datapoint['enc_word'] for datapoint in input_batch_list]
        enc_word_seq_lengths = torch.LongTensor(list(map(len, enc_word)))
        enc_max_seq_len = enc_word_seq_lengths.max()
        enc_word_seq_tensor = torch.zeros((batch_size, enc_max_seq_len)).long()

        enc_pos = [datapoint['enc_pos'] for datapoint in input_batch_list]
        enc_pos_tensor = torch.zeros((batch_size, enc_max_seq_len)).long()

        enc_mask = torch.zeros((batch_size, enc_max_seq_len)).byte()

        for idx, (seq, pos, seqlen) in enumerate(zip(enc_word, enc_pos, enc_word_seq_lengths)):
            enc_word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            enc_pos_tensor[idx, :seqlen] = torch.LongTensor(pos)

            enc_mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())

        enc_word_seq_lengths, enc_word_perm_idx = enc_word_seq_lengths.sort(0, descending=True)
        enc_word_seq_tensor = enc_word_seq_tensor[enc_word_perm_idx]
        enc_pos_tensor = enc_pos_tensor[enc_word_perm_idx]
        enc_mask = enc_mask[enc_word_perm_idx]

        _, enc_word_seq_recover = enc_word_perm_idx.sort(0, descending=False)

        enc_char = [datapoint['enc_char'] for datapoint in input_batch_list]
        enc_pad_chars = [enc_char[idx] + [[0]] * (enc_max_seq_len.item() - len(enc_char[idx])) for idx in range(len(enc_char))]
        enc_length_list = [list(map(len, pad_char)) for pad_char in enc_pad_chars]
        enc_max_word_len = max(list(map(max, enc_length_list)))
        enc_char_seq_tensor = torch.zeros((batch_size, enc_max_seq_len, enc_max_word_len)).long()
        enc_char_seq_lengths = torch.LongTensor(enc_length_list)
        for idx, (seq, seqlen) in enumerate(zip(enc_pad_chars, enc_char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                enc_char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        enc_char_seq_tensor = enc_char_seq_tensor[enc_word_perm_idx].view(batch_size * enc_max_seq_len.item(), -1)
        enc_char_seq_lengths = enc_char_seq_lengths[enc_word_perm_idx].view(batch_size * enc_max_seq_len.item(), )
        enc_char_seq_lengths, enc_char_perm_idx = enc_char_seq_lengths.sort(0, descending=True)
        enc_char_seq_tensor = enc_char_seq_tensor[enc_char_perm_idx]
        _, enc_char_seq_recover = enc_char_perm_idx.sort(0, descending=False)


        dec_word = [datapoint['dec_word'][:-1] for datapoint in input_batch_list]
        dec_word_seq_lengths = torch.LongTensor(list(map(len, dec_word)))
        dec_max_seq_len = dec_word_seq_lengths.max()
        dec_word_seq_tensor = torch.zeros((batch_size, dec_max_seq_len)).long()


        dec_mask = torch.zeros((batch_size, dec_max_seq_len)).byte()

        label = [datapoint['dec_word'][1:] for datapoint in input_batch_list]
        label_tensor = torch.zeros((batch_size, dec_max_seq_len)).long()
        # label length is the same as dec word length, also dec_mask


        for idx, (seq, l, seqlen) in enumerate(zip(dec_word, label, dec_word_seq_lengths)):
            dec_word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            dec_mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())
            label_tensor[idx, :seqlen] = torch.LongTensor(l)


        dec_word_seq_lengths, dec_word_perm_idx = dec_word_seq_lengths.sort(0, descending=True)
        dec_word_seq_tensor = dec_word_seq_tensor[dec_word_perm_idx]
        dec_mask = dec_mask[dec_word_perm_idx]
        label_tensor = label_tensor[dec_word_perm_idx]

        _, dec_word_seq_recover = dec_word_perm_idx.sort(0, descending=False)


        dec_char = [datapoint['dec_char'][:-1] for datapoint in input_batch_list]
        dec_pad_chars = [dec_char[idx] + [[0]] * (dec_max_seq_len.item() - len(dec_char[idx])) for idx in range(len(dec_char))]
        dec_length_list = [list(map(len, pad_char)) for pad_char in dec_pad_chars]
        dec_max_word_len = max(list(map(max, dec_length_list)))
        dec_char_seq_tensor = torch.zeros((batch_size, dec_max_seq_len, dec_max_word_len)).long()
        dec_char_seq_lengths = torch.LongTensor(dec_length_list)
        for idx, (seq, seqlen) in enumerate(zip(dec_pad_chars, dec_char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                dec_char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        dec_char_seq_tensor = dec_char_seq_tensor[enc_word_perm_idx].view(batch_size * dec_max_seq_len.item(), -1)
        dec_char_seq_lengths = dec_char_seq_lengths[enc_word_perm_idx].view(batch_size * dec_max_seq_len.item(), )
        dec_char_seq_lengths, dec_char_perm_idx = dec_char_seq_lengths.sort(0, descending=True)
        dec_char_seq_tensor = dec_char_seq_tensor[dec_char_perm_idx]
        _, dec_char_seq_recover = dec_char_perm_idx.sort(0, descending=False)

        if opt.gpu >= 0 and torch.cuda.is_available():
            enc_word_seq_tensor = enc_word_seq_tensor.cuda(opt.gpu)
            enc_word_seq_lengths = enc_word_seq_lengths.cuda(opt.gpu)
            enc_word_seq_recover = enc_word_seq_recover.cuda(opt.gpu)
            enc_pos_tensor = enc_pos_tensor.cuda(opt.gpu)
            enc_mask = enc_mask.cuda(opt.gpu)

            enc_char_seq_tensor = enc_char_seq_tensor(opt.gpu)
            enc_char_seq_lengths = enc_char_seq_lengths(opt.gpu)
            enc_char_seq_recover = enc_char_seq_recover(opt.gpu)

            dec_word_seq_tensor = dec_word_seq_tensor.cuda(opt.gpu)
            dec_word_seq_lengths = dec_word_seq_lengths.cuda(opt.gpu)
            dec_word_seq_recover = dec_word_seq_recover.cuda(opt.gpu)
            dec_mask = dec_mask.cuda(opt.gpu)
            label_tensor = label_tensor(opt.gpu)

            dec_char_seq_tensor = dec_char_seq_tensor(opt.gpu)
            dec_char_seq_lengths = dec_char_seq_lengths(opt.gpu)
            dec_char_seq_recover = dec_char_seq_recover(opt.gpu)


    return enc_word_seq_tensor, enc_word_seq_lengths, enc_word_seq_recover, enc_pos_tensor, enc_mask, \
           enc_char_seq_tensor, enc_char_seq_lengths, enc_char_seq_recover, dec_word_seq_tensor, dec_word_seq_lengths, \
           dec_word_seq_recover, dec_mask, label_tensor, dec_char_seq_tensor, dec_char_seq_lengths, dec_char_seq_recover


def pad(x, y):
    tokens = x

    lengths = [len(row) for row in tokens]
    max_len = max(lengths)

    tokens = pad_sequence(tokens, max_len)
    lengths = torch.LongTensor(lengths)

    y = torch.LongTensor(y).view(-1)


    return tokens, lengths, y


def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x