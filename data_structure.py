
class Entity:
    def __init__(self):
        self.id = ""
        self.type = ""

        self.spans = [] # a couple of spans, list (start, end)
        self.tkSpans = []
        self.labelSpans = []
        self.name = ""

        self.sent_idx = -1
        self.norm_ids = []
        self.norm_names = []
        self.norm_confidences = []


    def equals(self, other):

        if self.type == other.type and len(self.spans) == len(other.spans) :

            for i in range(len(self.spans)) :

                if self.spans[i][0] != other.spans[i][0] or self.spans[i][1] != other.spans[i][1]:
                    return False

            return True
        else:
            return False

    def equals_span(self, other):
        if len(self.spans) == len(other.spans):

            for i in range(len(self.spans)):

                if self.spans[i][0] != other.spans[i][0] or self.spans[i][1] != other.spans[i][1]:
                    return False

            return True

        else:
            return False

    def equalsTkSpan(self, other):
        if len(self.tkSpans) == len(other.tkSpans):

            for i in range(len(self.tkSpans)):

                if self.tkSpans[i][0] != other.tkSpans[i][0] or self.tkSpans[i][1] != other.tkSpans[i][1]:
                    return False

            return True

        else:
            return False



class Document:
    def __init__(self):
        self.entities = []
        self.sentences = []
        self.name = ""
        self.text = ""
        self.all_sents_inds = []





