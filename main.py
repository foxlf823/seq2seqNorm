import logging
from options import opt, config
import random
import numpy as np
import torch
from preprocess import load_data_pubtator, load_pmid, prepare_data, load_abbr


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


        train_datapoints = prepare_data(train_documents, abbr_dict)


        pass


