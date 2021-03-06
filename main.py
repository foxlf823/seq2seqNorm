import logging
from options import opt, config
import random
import numpy as np
import torch
from alphabet import Alphabet, build_alphabet, build_position_alphabet, datapoint2id, build_alphabet_1, datapoint2id_1
from train import train
from dictionary import load_ctd
from my_utils import makedir_and_clear


if __name__ == '__main__':

    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    if opt.random_seed != 0:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed_all(opt.random_seed)

    if opt.whattodo == 1: # ncbi
        from preprocess import load_data_pubtator, load_pmid, prepare_data, load_abbr, prepare_data_1, prepare_dict_data
        documents1 = load_data_pubtator(config['ncbi_dataset'])
        pmids = load_pmid(config['ncbi_trainid'])
        train_documents = []
        for document in documents1:
            if document.name in pmids:
                train_documents.append(document)


        pmids = load_pmid(config['ncbi_devid'])
        # dev_documents = []
        # for document in documents1:
        #     if document.name in pmids:
        #         dev_documents.append(document)
        for document in documents1:
            if document.name in pmids:
                train_documents.append(document)


        pmids = load_pmid(config['ncbi_testid'])
        # test_documents = []
        # for document in documents1:
        #     if document.name in pmids:
        #         test_documents.append(document)
        dev_documents = []
        for document in documents1:
            if document.name in pmids:
                dev_documents.append(document)
        test_documents = []


        abbr_dict = load_abbr(config['ncbi_abbr'])
        logging.info("loading dictionary ... ")
        dictionary = load_ctd(config['norm_dict'])

        logging.info("generate data points")
        train_datapoints = prepare_data(train_documents, abbr_dict, dictionary)
        dev_datapoints = prepare_data_1(dev_documents, abbr_dict, dictionary) # we use dev_datapoints and test_datapoints only for build alphabet
        if len(test_documents) != 0:
            test_datapoints = prepare_data_1(test_documents, abbr_dict, dictionary)
        if opt.pretraining:
            dict_datapoints = prepare_dict_data(dictionary)

        logging.info("build alphabet ...")
        enc_word_alphabet = Alphabet('enc_word')
        if opt.use_char:
            enc_char_alphabet = Alphabet('enc_char')
        else:
            enc_char_alphabet = None

        if opt.method == 'cla':
            dec_word_alphabet = None
            dec_char_alphabet = None
        else:
            dec_word_alphabet = Alphabet('dec_word')
            if opt.use_char:
                dec_char_alphabet = Alphabet('dec_char')
            else:
                dec_char_alphabet = None

            dec_word_alphabet.add('<SOS>')
            dec_word_alphabet.add('<EOS>')

        build_alphabet(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, train_datapoints)
        build_alphabet_1(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, dev_datapoints)
        if len(test_documents) != 0:
            build_alphabet_1(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, test_datapoints)
        if opt.pretraining:
            build_alphabet(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, dict_datapoints)

        if opt.method == 'cla':
            enc_word_alphabet.close()
            if opt.use_char:
                enc_char_alphabet.close()
            position_alphabet = None
        else:
            enc_word_alphabet.close()
            dec_word_alphabet.close()
            if opt.use_char:
                enc_char_alphabet.close()
                dec_char_alphabet.close()

            if opt.context == 'sent':
                position_alphabet = Alphabet('position')
                build_position_alphabet(position_alphabet)
                position_alphabet.close()
            else:
                position_alphabet = None

        logging.info("transfer data points to id")
        train_ids = datapoint2id(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
                                 train_datapoints, dictionary)
        dev_ids = datapoint2id_1(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
                                 dev_datapoints, dictionary)
        if len(test_documents) != 0:
            test_ids = datapoint2id_1(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
                                      test_datapoints, dictionary)
        else:
            test_ids = []

        if opt.pretraining:
            dict_ids = datapoint2id(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
                                    dict_datapoints, dictionary)
        else:
            dict_ids = []

        makedir_and_clear(opt.output)

        train(train_ids, dev_ids, test_ids, dict_ids, enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
              dictionary)





