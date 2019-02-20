import codecs
from fox_tokenizer import FoxTokenizer
from sortedcontainers import SortedSet
from my_utils import setList
from options import opt, config
import logging


class CTD_Dict():
    def __init__(self):
        self.name2id = {} # preferred name -> id
        self.id2name = {} # id -> CTD_Term
        self.altid2id = {} # alternative id -> id

    # given id, return prefered name (list of tokens)
    def getPreferName(self, id):
        if id in self.id2name:
            term = self.id2name.get(id)
            return term.preferred_name
        else:
            if id in self.altid2id:
                primary_id = self.altid2id[id]
                term = self.id2name.get(primary_id)
                return term.preferred_name
            else:
                raise RuntimeError("can't find id")

    # given name, return id
    def getID(self, name):
        if name in self.name2id:
            id_list = self.name2id.get(name)
            return id_list[0] # if one name corresponds to multiple ids, use the first
        else:
            return None





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

from preprocess import word_process
# this function returned a SortedSet, so it helps cleaning the dictionary.
def name_process(name, doc_abbr_dict):
    ret_words = SortedSet()

    tmp_tokens = FoxTokenizer.tokenize(0, name, True)
    for token in tmp_tokens:
        precessed_words = word_process(token, doc_abbr_dict)

        for pw in precessed_words:
            ret_words.add(pw)

    return ret_words

def load_ctd(file_path):

    dictionary = CTD_Dict()
    max_name_length = 0

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
            if len(columns) >= 8:
                Synonyms = columns[7]
            else:
                Synonyms = ""

            # if DiseaseID.find('MESH') == -1:
            #     raise RuntimeError("DiseaseID.find('MESH') == -1")

            DiseaseID = DiseaseID.split(':')[1]
            DiseaseName = name_process(DiseaseName, None)  # assume there are no abbr in the CTD
            if len(DiseaseName) == 0:
                raise RuntimeError("len(DiseaseName) == 0")
            else:
                if len(DiseaseName) > max_name_length:
                    max_name_length = len(DiseaseName)

            term = CTD_Term()
            for dn in DiseaseName:
                term.preferred_name.append(dn)

            key_name = tokenlist2key(term.preferred_name)
            if key_name in dictionary.name2id:
                # after preprocessing, one name may correspond to multiple id
                logging.debug("id {} key_name {} exists".format(DiseaseID, key_name))
                id_list = dictionary.name2id[key_name]
                id_list.append(DiseaseID)
            else:
                dictionary.name2id[key_name] = [DiseaseID]

            if len(Synonyms) != 0:
                for sm in Synonyms.split('|'):
                    ret_words = name_process(sm, None)
                    if len(ret_words) == 0:
                        # raise RuntimeError("len(ret_words) == 0")
                        continue
                    else:
                        if len(ret_words) > max_name_length:
                            max_name_length = len(ret_words)

                    sm_list = []
                    for rw in ret_words:
                        sm_list.append(rw)

                    if sm_list == term.preferred_name:
                        continue

                    setList(term.synonyms, sm_list)

            dictionary.id2name[DiseaseID] = term

            if AltDiseaseIDs != '':
                # if AltDiseaseIDs.find('OMIM') == -1:
                #     raise RuntimeError("AltDiseaseIDs.find('OMIM') == -1")

                alt_id_list = AltDiseaseIDs.split('|')
                for alt_id in alt_id_list:
                    alt_id = alt_id.split(':')[1]
                    if alt_id in dictionary.altid2id:
                        # we only consider one-to-one map of altid2id
                        logging.debug('alt_id {} already in dictionary.altid2id'.format(alt_id))
                    else:
                        dictionary.altid2id[alt_id] = DiseaseID

    logging.info("dictionary max_name_length: {}".format(max_name_length))

    return dictionary


if __name__ == '__main__':
    load_ctd('/Users/feili/project/DNorm-0.0.7/data/CTD_diseases_debug.tsv')

