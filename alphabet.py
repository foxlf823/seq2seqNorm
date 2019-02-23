
import json
import os
from options import opt


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        if not self.label:
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        
    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        if index == 0:
            if self.label:
                return self.instances[0]
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.instances[0]

    def size(self):
        # if self.label:
        #     return len(self.instances)
        # else:
        return len(self.instances) + 1

    def iteritems(self):
        # return self.instance2index.iteritems()
        return self.instance2index.items()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))


def build_alphabet(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, datapoints):
    for datapoint in datapoints:

        encoder_word = datapoint['enc_word']
        for word in encoder_word:
            enc_word_alphabet.add(word)

        if opt.use_char:
            encoder_char = datapoint['enc_char']
            for char_in_word in encoder_char:
                for ch in char_in_word:
                    enc_char_alphabet.add(ch)

        if dec_word_alphabet is not None:
            decoder_word = datapoint['dec_word']
            for word in decoder_word:
                dec_word_alphabet.add(word)

        if dec_char_alphabet is not None:
            if opt.use_char:
                decoder_char = datapoint['dec_char']
                for char_in_word in decoder_char:
                    for ch in char_in_word:
                        dec_char_alphabet.add(ch)




def build_alphabet_1(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, datapoints):
    for datapoint_for_one_doc in datapoints:

        for datapoint in datapoint_for_one_doc:

            encoder_word = datapoint['enc_word']
            for word in encoder_word:
                enc_word_alphabet.add(word)

            if opt.use_char:
                encoder_char = datapoint['enc_char']
                for char_in_word in encoder_char:
                    for ch in char_in_word:
                        enc_char_alphabet.add(ch)

            if dec_word_alphabet is not None:
                decoder_word = datapoint['dec_word']
                for word in decoder_word:
                    dec_word_alphabet.add(word)

            if dec_char_alphabet is not None:
                if opt.use_char:
                    decoder_char = datapoint['dec_char']
                    for char_in_word in decoder_char:
                        for ch in char_in_word:
                            dec_char_alphabet.add(ch)



def build_position_alphabet(alphabet):
    # we assume the sentence is not longer than this
    for i in range(1000):
        alphabet.add(i)
        alphabet.add(-i)

def datapoint2id(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
                 datapoints, dictionary):
    ids = []
    for datapoint in datapoints:
        id_dict = {}

        encoder_word = datapoint['enc_word']
        encoder_word_id = []
        for word in encoder_word:
            encoder_word_id.append(enc_word_alphabet.get_index(word))
        id_dict['enc_word'] = encoder_word_id

        if opt.use_char:
            encoder_char = datapoint['enc_char']
            encoder_char_id = []
            for char_in_word in encoder_char:
                char_in_word_id = []
                for ch in char_in_word:
                    char_in_word_id.append(enc_char_alphabet.get_index(ch))
                encoder_char_id.append(char_in_word_id)
            id_dict['enc_char'] = encoder_char_id


        encoder_position = datapoint['enc_pos']
        encoder_position_id = []
        for position in encoder_position:
            encoder_position_id.append(position_alphabet.get_index(position))
        id_dict['enc_pos'] = encoder_position_id

        if opt.method == 'cla':
            id_dict['dec_id'] = dictionary.id_alphabet.get_index(datapoint['dec_id'])

        else:
            decoder_word = datapoint['dec_word']
            decoder_word_id = []
            for word in decoder_word:
                decoder_word_id.append(dec_word_alphabet.get_index(word))
            id_dict['dec_word'] = decoder_word_id

            if opt.use_char:
                decoder_char = datapoint['dec_char']
                decoder_char_id = []
                for char_in_word in decoder_char:
                    char_in_word_id = []
                    for ch in char_in_word:
                        char_in_word_id.append(dec_char_alphabet.get_index(ch))
                    decoder_char_id.append(char_in_word_id)
                id_dict['dec_char'] = decoder_char_id

        ids.append(id_dict)

    return ids

def datapoint2id_1(enc_word_alphabet, enc_char_alphabet, dec_word_alphabet, dec_char_alphabet, position_alphabet,
                   datapoints, dictionary):
    ids = []
    for datapoint_for_one_doc in datapoints:
        ids_for_one_doc = []
        for datapoint in datapoint_for_one_doc:
            id_dict = {}

            encoder_word = datapoint['enc_word']
            encoder_word_id = []
            for word in encoder_word:
                encoder_word_id.append(enc_word_alphabet.get_index(word))
            id_dict['enc_word'] = encoder_word_id

            if opt.use_char:
                encoder_char = datapoint['enc_char']
                encoder_char_id = []
                for char_in_word in encoder_char:
                    char_in_word_id = []
                    for ch in char_in_word:
                        char_in_word_id.append(enc_char_alphabet.get_index(ch))
                    encoder_char_id.append(char_in_word_id)
                id_dict['enc_char'] = encoder_char_id

            encoder_position = datapoint['enc_pos']
            encoder_position_id = []
            for position in encoder_position:
                encoder_position_id.append(position_alphabet.get_index(position))
            id_dict['enc_pos'] = encoder_position_id

            if opt.method == 'cla':
                id_dict['dec_id'] = dictionary.id_alphabet.get_index(datapoint['dec_id'])

            else:

                decoder_word = datapoint['dec_word']
                decoder_word_id = []
                for word in decoder_word:
                    decoder_word_id.append(dec_word_alphabet.get_index(word))
                id_dict['dec_word'] = decoder_word_id

                if opt.use_char:
                    decoder_char = datapoint['dec_char']
                    decoder_char_id = []
                    for char_in_word in decoder_char:
                        char_in_word_id = []
                        for ch in char_in_word:
                            char_in_word_id.append(dec_char_alphabet.get_index(ch))
                        decoder_char_id.append(char_in_word_id)
                    id_dict['dec_char'] = decoder_char_id

            ids_for_one_doc.append(id_dict)

        ids.append(ids_for_one_doc)

    return ids

