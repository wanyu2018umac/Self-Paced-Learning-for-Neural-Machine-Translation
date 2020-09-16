import math
import numpy


class BLEU:
    def __init__(self):
        return

    def brevity_penalty(self, can_length: int, ref_length: int):
        if can_length == 0:
            return 0

        if can_length >= ref_length:
            return 1

        return math.exp(1 - ref_length / can_length)

    def n_gram_calculate(self, tokens: list, n: int = 4):
        n_grams = []

        for i in range(0, n):
            n_grams.append(self.n_gram_derive(tokens, i))

        return n_grams

    def n_gram_derive(self, tokens: list, n: int):
        dict = {}

        for i in range(0, len(tokens) - n):
            n_gram = tuple(tokens[i: i + n + 1])

            if n_gram in dict.keys():
                dict[n_gram] += 1
            else:
                dict[n_gram] = 1

        return dict

    def n_gram_matches(self, candidate: dict, reference: dict):
        matched_grams = set(candidate.keys()).intersection(set(reference.keys()))
        matched_grams_num = sum(min(candidate.get(gram, 0), reference.get(gram, 0)) for gram in matched_grams)
        all_grams_num = sum(candidate.values())
        return matched_grams_num, all_grams_num

    def sentence_score(self, candidate: list, reference: [list], n: int = 4):
        can_grams = self.n_gram_calculate(candidate, n)
        ref_grams_list = list(self.n_gram_calculate(d, n) for d in reference)
        ref_grams_set = list(set(sum([list(ref[i]) for ref in ref_grams_list], list())) for i in range(0, n))
        ref_grams = list(dict(zip(ref_grams_set[i],
                                  map(lambda gram: max(l[i].get(gram, 0) for l in ref_grams_list), ref_grams_set[i])))
                         for i in range(0, n))
        matched_grams = numpy.zeros(n)
        all_grams = numpy.zeros(n)

        for i in range(0, n):
            matched_grams_step, all_grams_step = self.n_gram_matches(can_grams[i], ref_grams[i])
            matched_grams[i] += matched_grams_step
            all_grams[i] += all_grams_step

        return matched_grams, all_grams

    def document_score(self, candidate: [list], reference: [[list]], n: int = 4):
        all_stats = list(self.sentence_score(can, ref, n)
                         for can, ref in zip(candidate, list(list(ref[idx] for ref in reference)
                                                             for idx in range(0, len(candidate)))))
        matched_grams_stats_container = list(s[0] for s in all_stats)
        all_grams_stats_container = list(s[1] for s in all_stats)

        matched_grams_stats = sum(matched_grams_stats_container, numpy.zeros(n))
        all_grams_stats = sum(all_grams_stats_container, numpy.zeros(n))

        return matched_grams_stats, all_grams_stats

    def print_score_stats(self, matched_grams_stats, all_grams_stats,
                          can_length: int, ref_length: int, brevity_penalty: float,
                          n: int = 4,
                          weight: numpy.array = numpy.array(0.25).repeat(4)):
        print('%15s\t%s' % ('N-grams', '\t'.join('%8d' % x for x in range(0, n))))
        print('%15s\t%s' % ('Weight', '\t'.join('%8.6f' % x for x in weight)))

        print('%15s\t%s' % ('Matched grams', '\t'.join('%8d' % int(x) for x in matched_grams_stats)))
        print('%15s\t%s' % ('All grams', '\t'.join('%8d' % int(x) for x in all_grams_stats)))

        scores = numpy.log(matched_grams_stats / all_grams_stats)
        scores[scores != scores] = -float('inf')
        print('%15s\t%s' % ('Scores', '\t'.join('%7.5f' % x for x in scores)))

        print('Penalty: %.4f, (can: %d, ref: %d)' % (brevity_penalty, can_length, ref_length))
        return brevity_penalty * numpy.exp(scores.dot(weight))

    def bleu(self, candidate: [list], reference: [[list]], n: int = 4,
             weight: numpy.array = numpy.array(0.25).repeat(4)):
        matched_grams_stats, all_grams_stats = self.document_score(candidate, reference, n)
        can_length, ref_length = self.get_closest_lengths(candidate, reference)
        bp = self.brevity_penalty(can_length, ref_length)
        score = self.print_score_stats(matched_grams_stats, all_grams_stats, can_length, ref_length, bp, n, weight)

        return score

    def get_closest_lengths(self, candidate: [list], reference: [[list]]):
        num_of_examples = len(candidate)
        can_lengths = numpy.array(list(len(x) for x in candidate))
        ref_lengths = list(list(len(x) for x in ref) for ref in reference)

        ref_lengths_out = numpy.array(list(self.get_closest_length(can_lengths[i],
                                                                   list(ref_length[i] for ref_length in ref_lengths))
                                           for i in range(0, num_of_examples)))

        return can_lengths.sum(), ref_lengths_out.sum()

    def get_closest_length(self, candidate: int, reference: [int]):
        origin_diff = numpy.array(reference) - candidate
        abs_diff = numpy.abs(origin_diff)
        sorted_idxs = abs_diff.argsort()
        min_diff = abs_diff[sorted_idxs[0]]

        num_of_chosen = sum((min_diff == abs_diff))

        if num_of_chosen == 1:
            return reference[sorted_idxs[0]]

        for idx in range(0, num_of_chosen):
            if origin_diff[idx] < 0:
                return reference[idx]

        return reference[sorted_idxs[0]]
