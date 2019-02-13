import logging
from options import opt, config
import random
import numpy as np
import torch
from alphabet import Alphabet, build_alphabet, build_position_alphabet, datapoint2id, build_alphabet_1, datapoint2id_1
from torch.utils.data import DataLoader
from dataload import MyDataset, my_collate


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
        from preprocess import load_data_pubtator, load_pmid, prepare_data, load_abbr, prepare_data_1
        documents1 = load_data_pubtator(config['ncbi_dataset'])
        pmids = load_pmid(config['ncbi_trainid'])
        train_documents = []
        for document in documents1:
            if document.name in pmids:
                train_documents.append(document)


        pmids = load_pmid(config['ncbi_devid'])
        dev_documents = []
        for document in documents1:
            if document.name in pmids:
                dev_documents.append(document)


        pmids = load_pmid(config['ncbi_testid'])
        test_documents = []
        for document in documents1:
            if document.name in pmids:
                test_documents.append(document)


        abbr_dict = load_abbr(config['ncbi_abbr'])

        logging.info("generate data points")
        train_datapoints = prepare_data(train_documents, abbr_dict)
        dev_datapoints = prepare_data_1(dev_documents, abbr_dict) # we use dev_datapoints and test_datapoints only for build alphabet
        test_datapoints = prepare_data_1(test_documents, abbr_dict)

        logging.info("build alphabet ...")
        word_alphabet = Alphabet('word')
        if opt.use_char:
            char_alphabet = Alphabet('char')
        else:
            char_alphabet = None
        word_alphabet.add('<SOS>')
        word_alphabet.add('<EOS>')
        build_alphabet(word_alphabet, char_alphabet, train_datapoints)
        build_alphabet_1(word_alphabet, char_alphabet, dev_datapoints)
        build_alphabet_1(word_alphabet, char_alphabet, test_datapoints)
        word_alphabet.close()


        position_alphabet = Alphabet('position')
        build_position_alphabet(position_alphabet)
        position_alphabet.close()

        logging.info("transfer data points to id")
        train_ids = datapoint2id(word_alphabet, char_alphabet, position_alphabet, train_datapoints)
        dev_ids = datapoint2id_1(word_alphabet, char_alphabet, position_alphabet, dev_datapoints)
        test_ids = datapoint2id_1(word_alphabet, char_alphabet, position_alphabet, test_datapoints)

        train_loader = DataLoader(MyDataset(train_ids), opt.batch_size, shuffle=True, collate_fn=my_collate)





