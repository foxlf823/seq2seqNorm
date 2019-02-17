from torch.utils.data import DataLoader
from dataload import MyDataset, my_collate_1
from options import opt, config
from embedding import initialize_emb
from model import Encoder, Decoder
import torch.optim as optim
import logging
import time
import itertools
import torch

def train(train_ids, dev_ids, test_ids, enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet):

    enc_word_emb = initialize_emb(config.get('word_emb'), enc_word_alphabet, opt.word_emb_dim)
    pos_emb = initialize_emb(None, position_alphabet, opt.pos_emb_dim)
    if opt.use_char:
        enc_char_emb = initialize_emb(config.get('char_emb'), enc_char_alphabet, opt.char_emb_dim)
    else:
        enc_char_emb = None

    encoder = Encoder(enc_word_emb, pos_emb, enc_char_emb)

    dec_word_emb = initialize_emb(config.get('word_emb'), dec_word_alphabet, opt.word_emb_dim)
    if opt.use_char:
        dec_char_emb = initialize_emb(config.get('char_emb'), dec_char_alphabet, opt.char_emb_dim)
    else:
        dec_char_emb = None

    decoder = Decoder(dec_word_emb, dec_char_emb, dec_word_alphabet)

    # train_loader = DataLoader(MyDataset(train_ids), opt.batch_size, shuffle=True, collate_fn=my_collate)
    if opt.batch_size != 1:
        raise RuntimeError("currently, only support batch size 1")
    train_loader = DataLoader(MyDataset(train_ids), opt.batch_size, shuffle=True, collate_fn=my_collate_1)

    optimizer = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, weight_decay=opt.l2)

    if opt.tune_wordemb == False:
        encoder.free_emb()
        decoder.free_emb()

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")

    for idx in range(opt.iter):
        epoch_start = time.time()

        encoder.train()
        decoder.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        sum_loss = 0

        correct, total = 0, 0

        for i in range(num_iter):
            enc_word_seq_tensor, enc_pos_tensor, \
            enc_char_seq_tensor, enc_char_seq_lengths, enc_char_seq_recover, dec_word_seq_tensor, \
            label_tensor, dec_char_seq_tensor, dec_char_seq_lengths, dec_char_seq_recover = next(train_iter)

            encoder_outputs, encoder_hidden = encoder.forward(enc_word_seq_tensor, enc_pos_tensor, \
                enc_char_seq_tensor, enc_char_seq_lengths, enc_char_seq_recover)

            if opt.use_teacher_forcing:
                loss, score = decoder.forward_teacher_forcing(encoder_outputs, encoder_hidden,
                                                              dec_word_seq_tensor, label_tensor,
                                                              dec_char_seq_tensor, dec_char_seq_lengths, dec_char_seq_recover)
            else:
                raise RuntimeError("currently, we don't support non-teacher-forcing training")

            sum_loss += loss.item()

            loss.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.gradient_clip)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.gradient_clip)
            optimizer.step()
            encoder.zero_grad()
            decoder.zero_grad()

            total += label_tensor.size(0)*label_tensor.size(1)
            _, pred = torch.max(score, 1)
            correct += (pred == label_tensor.view(-1)).sum().item()

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
            idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))








