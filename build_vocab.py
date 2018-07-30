import nltk
import pickle
import argparse
from collections import Counter
from vist import VIST

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(sis_file, threshold):
    vist = VIST(sis_file, )
    counter = Counter()

    ids = vist.stories.keys()
    for i, id in enumerate(ids):
        story = vist.stories[id]
        for annotation in story:
            caption = annotation['text']
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
            except Exception:
                pass
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the story captions." %(i, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

def main(args):
    vocab = build_vocab(sis_file=args.sis_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sis_path', type=str,
                        default='./data/sis/train.story-in-sequence.json',
                        help='path for train sis file')
    parser.add_argument('--vocab_path', type=str, default='./models/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
