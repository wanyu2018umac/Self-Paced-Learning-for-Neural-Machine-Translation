class BytePairEncoding:
    def __init__(self, bpe_suffix_token: str, vocab: dict):
        self.bpe_suffix_token = bpe_suffix_token
        self.vocab = vocab

        return

    def subwords2words(self, subwords: [str]):
        return ' '.join(subwords).replace(self.bpe_suffix_token + ' ', '').replace(self.bpe_suffix_token, '').split()

    def words2subwords(self, words: [str]):
        output = []

        for word in words:
            if word in self.vocab.keys():
                output.append(word)
            else:
                handling = word
                pivot = len(word)

                while handling != '':
                    pivot -= 1
                    if pivot == 0:
                        output.append(handling)
                        break

                    if handling[:pivot] + self.bpe_suffix_token in self.vocab.keys():
                        output.append(handling[:pivot] + self.bpe_suffix_token)
                        handling = handling[pivot:]

                        if handling in self.vocab.keys():
                            output.append(handling)
                            break

                        pivot = len(handling)

        return output
