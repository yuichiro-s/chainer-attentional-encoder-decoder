IGNORE_ID = -1
BOS_ID = 0
EOS_ID = 1
UNK_ID = 2
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


class Vocab(object):
    """Mapping between words and IDs."""

    def __init__(self):
        self.i2w = []
        self.w2i = {}
        self.add_word(BOS)
        self.add_word(EOS)
        self.add_word(UNK)
        assert BOS_ID == self.get_id(BOS)
        assert EOS_ID == self.get_id(EOS)
        assert UNK_ID == self.get_id(UNK)

    def add_word(self, word):
        assert word not in self.w2i, word
        new_id = self.size()
        self.i2w.append(word)
        self.w2i[word] = new_id

    def get_id(self, word):
        return self.w2i.get(word, UNK_ID)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for w in self.i2w:
                print(w, file=f)

    @classmethod
    def load(cls, path, size=-1):
        vocab = Vocab()
        with open(path) as f:
            for i, line in enumerate(f):
                if size >= 0 and vocab.size() >= size:
                    break
                w = line.strip().split('\t')[0]
                if i == BOS_ID:
                    assert w == BOS
                elif i == EOS_ID:
                    assert w == EOS
                elif i == UNK_ID:
                    assert w == UNK
                else:
                    vocab.add_word(w)
                    assert vocab.get_id(w) == i
        return vocab
