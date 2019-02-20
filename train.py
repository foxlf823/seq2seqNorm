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
import os

def train(train_ids, dev_ids, test_ids, dict_ids, enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
          dictionary):

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

    if opt.pretraining:
        dict_loader = DataLoader(MyDataset(dict_ids), opt.batch_size, shuffle=True, collate_fn=my_collate_1)

        logging.info("start dict pretraining ...")
        logging.info("dict pretraining datapoints: {}".format(len(dict_ids)))

        bad_counter = 0
        best_accuracy = 0

        for idx in range(9999):
            epoch_start = time.time()

            encoder.train()
            decoder.train()

            correct, total = 0, 0

            sum_loss = 0

            train_iter = iter(dict_loader)
            num_iter = len(dict_loader)

            for i in range(num_iter):
                enc_word_seq_tensor, enc_pos_tensor, \
                enc_char_seq_tensor, enc_char_seq_lengths, enc_char_seq_recover, dec_word_seq_tensor, \
                label_tensor, dec_char_seq_tensor = next(train_iter)

                encoder_outputs, encoder_hidden = encoder.forward(enc_word_seq_tensor, enc_pos_tensor, \
                                                                  enc_char_seq_tensor, enc_char_seq_lengths,
                                                                  enc_char_seq_recover)

                loss, total_this_batch, correct_this_batch = decoder.forward_train(encoder_outputs, encoder_hidden,
                                                                                   dec_word_seq_tensor, label_tensor,
                                                                                   dec_char_seq_tensor)

                sum_loss += loss.item()

                loss.backward()

                if opt.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.gradient_clip)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.gradient_clip)
                optimizer.step()
                encoder.zero_grad()
                decoder.zero_grad()

                total += total_this_batch
                correct += correct_this_batch

            epoch_finish = time.time()
            accuracy = 100.0 * correct / total
            logging.info("epoch: %s pretraining finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
                idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

            if accuracy > opt.expected_accuracy:
                logging.info("Exceed expected training accuracy, breaking ... ")
                break

            if accuracy > best_accuracy:
                logging.info("Exceed previous best accuracy: %.2f" % (best_accuracy))
                best_accuracy = accuracy

                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter >= opt.patience:
                logging.info('Pretraining Early Stop!')
                break


    best_dev_f = -10

    bad_counter = 0

    logging.info("start training ...")
    logging.info("training datapoints: {}".format(len(train_ids)))
    if dev_ids is not None and len(dev_ids) != 0:
        logging.info("dev datapoints: {}".format(len(dev_ids)))
    if test_ids is not None and len(test_ids) != 0:
        logging.info("test datapoints: {}".format(len(test_ids)))

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
            label_tensor, dec_char_seq_tensor = next(train_iter)

            encoder_outputs, encoder_hidden = encoder.forward(enc_word_seq_tensor, enc_pos_tensor, \
                enc_char_seq_tensor, enc_char_seq_lengths, enc_char_seq_recover)


            loss, total_this_batch, correct_this_batch = decoder.forward_train(encoder_outputs, encoder_hidden,
                                                              dec_word_seq_tensor, label_tensor,
                                                              dec_char_seq_tensor)

            sum_loss += loss.item()

            loss.backward()

            if opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.gradient_clip)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.gradient_clip)
            optimizer.step()
            encoder.zero_grad()
            decoder.zero_grad()

            total += total_this_batch
            correct += correct_this_batch

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
            idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))


        if dev_ids is not None and len(dev_ids) != 0:
            p, r, f = evaluate(dev_ids, encoder, decoder, dec_word_alphabet, dec_char_alphabet, dictionary)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            best_dev_f = f

            bad_counter = 0

            torch.save(encoder, os.path.join(opt.output, "encoder.pkl"))
            torch.save(decoder, os.path.join(opt.output, "decoder.pkl"))
            torch.save(enc_word_alphabet, os.path.join(opt.output, "enc_word_alphabet.pkl"))
            torch.save(enc_char_alphabet, os.path.join(opt.output, "enc_char_alphabet.pkl"))
            torch.save(dec_word_alphabet, os.path.join(opt.output, "dec_word_alphabet.pkl"))
            torch.save(dec_char_alphabet, os.path.join(opt.output, "dec_char_alphabet.pkl"))
            torch.save(position_alphabet, os.path.join(opt.output, "position_alphabet.pkl"))

            if test_ids is not None and len(test_ids) != 0:
                p, r, f = evaluate(test_ids, encoder, decoder, dec_word_alphabet, dec_char_alphabet, dictionary)
                logging.info("Test: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))

        else:
            bad_counter += 1

        if bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info("train finished")



# evaluation method: for each document, we compute p,r,f1 using ID.
# predicted id set vs. gold id set
def evaluate(datapoints, encoder, decoder, dec_word_alphabet, dec_char_alphabet, dictionary):
    encoder.eval()
    decoder.eval()

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    for datapoint_for_one_doc in datapoints:

        gold_id_set = set()

        predict_id_set = set()

        for datapoint in datapoint_for_one_doc:
            # get gold id
            gold_idx = datapoint['dec_word'][1:-1]
            gold_token = []
            for gold_idx_ in gold_idx:
                token = dec_word_alphabet.get_instance(gold_idx_)
                gold_token.append(token)
            gold_names = tokenlist2key_1(gold_token)
            for gold_name in gold_names:
                gold_id = dictionary.getID(gold_name)
                if gold_id is None:
                    raise RuntimeError("gold_id is None")
                gold_id_set.add(gold_id)
            # get predict id
            enc_word_seq_tensor, enc_pos_tensor, \
            enc_char_seq_tensor, enc_char_seq_lengths, enc_char_seq_recover, dec_word_seq_tensor, \
            label_tensor, dec_char_seq_tensor = my_collate_1([datapoint])

            encoder_outputs, encoder_hidden = encoder.forward(enc_word_seq_tensor, enc_pos_tensor, \
                                                              enc_char_seq_tensor, enc_char_seq_lengths,
                                                              enc_char_seq_recover)

            predict_token = decoder.forward_infer(encoder_outputs, encoder_hidden, dec_word_alphabet, dec_char_alphabet)

            predict_names = tokenlist2key_1(predict_token)
            for predict_name in predict_names:
                predict_id = dictionary.getID(predict_name)
                if predict_id is None: # if predict id is none, this doesn't count as errors
                    pass
                else:
                    predict_id_set.add(predict_id)


        ct_gold += len(gold_id_set)
        ct_predicted += len(predict_id_set)
        ct_correct += len(gold_id_set & predict_id_set)




    if ct_gold == 0 or ct_predicted == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_correct * 1.0 / ct_predicted
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure


def tokenlist2key_1(token_list):
    rets = []
    ret = ""
    start = True
    for i, token in enumerate(token_list):

        if token == '|':  # composite entity
            rets.append(ret)
            ret = ""
            start = True
        else:
            if start:
                ret += token
                start = False
            else:
                ret += "_"+token

    rets.append(ret)

    return rets

