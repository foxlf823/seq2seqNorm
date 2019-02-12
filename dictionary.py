import codecs
from fox_tokenizer import FoxTokenizer
from sortedcontainers import SortedSet
from preprocess import word_process
from my_utils import setList


# this function returned a SortedSet, so it helps cleaning the dictionary.
def name_process(name, doc_abbr_dict):
    ret_words = SortedSet()

    tmp_tokens = FoxTokenizer.tokenize(0, name, True)
    for token in tmp_tokens:
        precessed_words = word_process(token, doc_abbr_dict)

        for pw in precessed_words:
            ret_words.add(pw)

    return ret_words

class CTD_Dict():
    def __init__(self):
        self.name2id = {} # preferred name -> id
        self.id2name = {} # id -> CTD_Term

class CTD_Term():
    def __init__(self):
        self.preferred_name = [] # list
        self.synonyms = [] # list list

def tokenlist2key(token_list):
    ret = ""
    for i, token in enumerate(token_list):
        if i == len(token_list)-1:
            ret += token
        else:
            ret += token+"_"

    return ret


def load_ctd(file_path):

    dictionary = CTD_Dict()

    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line == u'':
                continue

            if line[0] == '#':
                continue

            columns = line.split("\t")

            DiseaseName = columns[0]
            DiseaseID = columns[1]
            AltDiseaseIDs = columns[2]
            Definition = columns[3]
            ParentIDs = columns[4]
            TreeNumbers = columns[5]
            ParentTreeNumbers = columns[6]
            Synonyms = columns[7]

            DiseaseID = columns[1].split(':')[1]
            DiseaseName = name_process(DiseaseName, None)  # assume there are no abbr in the CTD
            if len(DiseaseName) == 0:
                raise RuntimeError("len(DiseaseName) == 0")

            term = CTD_Term()
            for dn in DiseaseName:
                term.preferred_name.append(dn)

            key_name = tokenlist2key(term.preferred_name)
            if key_name in dictionary.name2id:
                raise RuntimeError('key_name in dictionary.name2id')
            else:
                dictionary.name2id[key_name] = DiseaseID

            for sm in Synonyms.split('|'):
                ret_words = name_process(sm, None)
                if len(ret_words) == 0:
                    raise RuntimeError("len(ret_words) == 0")
                sm_list = []
                for rw in ret_words:
                    sm_list.append(rw)

                if sm_list == term.preferred_name:
                    continue

                setList(term.synonyms, sm_list)

            dictionary.id2name[DiseaseID] = term



    return dictionary






if __name__ == '__main__':
    load_ctd('/Users/feili/project/DNorm-0.0.7/data/CTD_diseases_debug.tsv')

