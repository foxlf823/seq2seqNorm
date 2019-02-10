
import codecs
from data_structure import Document, Entity
from fox_tokenizer import FoxTokenizer
import nltk
nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
# from nltk.stem import WordNetLemmatizer
# wnl = WordNetLemmatizer()
import logging
from my_utils import setMapMap
from stopword import stop_word

def load_data_pubtator(file_path):

    # stat
    ct_doc = 0
    ct_entity = 0

    documents = []
    with codecs.open(file_path, 'r', 'UTF-8') as fp:

        document = None

        for line in fp:

            line = line.strip()

            if line == '':
                if document is None:
                    continue
                else:
                    # save the document
                    documents.append(document)
                    document = None
                    ct_doc += 1
            elif line.find('|t|') != -1:
                # a new document
                document = Document()
                columns = line.split('|t|')
                document.name = columns[0]
                document.text = columns[1] + " " # offset need + 1


            elif line.find('|a|') != -1:

                columns = line.split('|a|')

                document.text += columns[1]

                generator = nlp_tool.span_tokenize(document.text)
                for t in generator:
                    document.all_sents_inds.append(t)


                for ind in range(len(document.all_sents_inds)):
                    t_start = document.all_sents_inds[ind][0]
                    t_end = document.all_sents_inds[ind][1]

                    tmp_tokens = FoxTokenizer.tokenize(t_start, document.text[t_start:t_end], False)
                    sentence_tokens = []
                    for token_idx, token in enumerate(tmp_tokens):
                        token_dict = {}
                        token_dict['start'], token_dict['end'] = token[1], token[2]
                        token_dict['text'] = token[0]

                        sentence_tokens.append(token_dict)

                    document.sentences.append(sentence_tokens)

            else:
                columns = line.split('\t')

                if columns[1] == 'CID': # for cdr corpus, we ignore relation
                    continue

                if columns[4].find("Chemical") != -1: # for cdr corpus, we ignore chemical
                    continue

                entity = Entity()
                entity.spans.append([int(columns[1]), int(columns[2])])
                entity.name = columns[3]
                entity.type = columns[4]


                ids = columns[5].split('|')
                for id in ids:
                    if id == '-1':
                        raise RuntimeError("id == -1")
                    entity.norm_ids.append(id)

                # columns[6], cdr may has Individual mentions, we don't use it yet

                for sent_idx, (sent_start, sent_end) in enumerate(document.all_sents_inds):
                    if entity.spans[0][0] >= sent_start and entity.spans[0][1] <= sent_end: # we assume entity has only one span
                        entity.sent_idx = sent_idx
                        break
                if entity.sent_idx == -1:
                    raise RuntimeError("can't find entity.sent_idx")

                document.entities.append(entity)
                ct_entity += 1

    logging.info("document number {}, entity number {}".format(ct_doc, ct_entity))


    return documents

def load_pmid(file_path): # for ncbi disease corpus
    documents = set()
    with codecs.open(file_path, 'r', 'UTF-8') as fp:

        for line in fp:
            line = line.strip()

            if line != '':
                documents.add(line)

    return documents

def load_abbr(file_path):
    abbr_dict = {}

    with codecs.open(file_path, 'r', 'UTF-8') as fp:

        for line in fp:
            line = line.strip()

            if line != '':
                columns = line.split("\t")

                setMapMap(abbr_dict, columns[0], columns[1], columns[2])

    return abbr_dict

# abbr replaced, stopword removed, keep number, lower,
# return 0, 1 or multiple words
def word_process(word, doc_abbr_dict):
    ret_words = []
    full_name = None
    if doc_abbr_dict is not None:
        full_name_list = doc_abbr_dict.get(word)
        if full_name_list is not None:

            if len(full_name_list) != 1:
                raise RuntimeError("full_name_list is not 1")

            full_name = FoxTokenizer.tokenize(0, full_name_list[0], True)


    if full_name is not None:

        for w in full_name:
            w = w.lower()

            if w in stop_word:
                continue

            # lemma
            # w = wnl.lemmatize(w)

            ret_words.append(w)

    else:
        w = word.lower()
        if w not in stop_word:
            # lemma
            # w = wnl.lemmatize(w)

            ret_words.append(w)

    return ret_words

def prepare_data_for_one_document(document, abbr_dict):
    datapoints = []
    doc_abbr_dict = abbr_dict.get(document.name) # doc_abbr_dict may be none

    for entity in document.entities:

        datapoint = {} # one entity is a datapoint

        encoder_sent = []

        original_sentence = document.sentences[entity.sent_idx]

        for token_dict in original_sentence:
            start = token_dict['start']
            end = token_dict['end']
            word = token_dict['text']
            precessed_words = word_process(word, doc_abbr_dict)

            for pw in precessed_words:
                encoder_sent.append(pw)

        pass



def prepare_data(documents, abbr_dict):
    datapoints = []
    for document in documents:
        datapoints_for_one_doc = prepare_data_for_one_document(document, abbr_dict)
        datapoints.extend(datapoints_for_one_doc)

    return datapoints




if __name__ == '__main__':


    # documents1 = load_data_pubtator('/Users/feili/dataset/NCBI_disease_corpus/NCBItrainset_corpus.txt')
    # print(len(documents1))
    # documents1 = load_data_pubtator('/Users/feili/dataset/NCBI_disease_corpus/NCBIdevelopset_corpus.txt')
    # print(len(documents1))
    # documents1 = load_data_pubtator('/Users/feili/dataset/NCBI_disease_corpus/NCBItestset_corpus.txt')
    # print(len(documents1))

    documents1 = load_data_pubtator('/Users/feili/project/DNorm-0.0.7/data/NCBI_disease/Corpus.txt')
    pmids = load_pmid('/Users/feili/project/DNorm-0.0.7/data/NCBI_disease/NCBI_corpus_training_PMIDs.txt')
    documents = []
    for document in documents1:
        if document.name in pmids:
            documents.append(document)
    print(len(documents))

    pmids = load_pmid('/Users/feili/project/DNorm-0.0.7/data/NCBI_disease/NCBI_corpus_development_PMIDs.txt')
    documents = []
    for document in documents1:
        if document.name in pmids:
            documents.append(document)
    print(len(documents))

    pmids = load_pmid('/Users/feili/project/DNorm-0.0.7/data/NCBI_disease/NCBI_corpus_test_PMIDs.txt')
    documents = []
    for document in documents1:
        if document.name in pmids:
            documents.append(document)
    print(len(documents))

    # documents1 = load_data_pubtator('/Users/feili/dataset/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt')
    # print(len(documents1))
    # documents1 = load_data_pubtator('/Users/feili/dataset/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt')
    # print(len(documents1))
    # documents1 = load_data_pubtator('/Users/feili/dataset/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt')
    # print(len(documents1))

    pass