import numpy
import torch


class SearchNode:
    def __init__(self, max_seq_length: int, eos_idx: int,
                 decoding_alpha: float,
                 device: torch.device):
        self.max_seq_length = max_seq_length
        self.device = device
        self.route = numpy.zeros([self.max_seq_length], dtype=int)
        self.probability_score = numpy.zeros([self.max_seq_length])
        self.steps = 0
        self.final_score = 0.0
        self.eos_idx = eos_idx
        self.eos_updated = False

        self.decoding_alpha = decoding_alpha

        return

    def update(self, route_add, score_add):
        route_add = int(route_add)
        score_add = float(score_add)

        if self.eos_updated:
            return

        if route_add == self.eos_idx:
            self.eos_updated = True
            return

        self.route[self.steps] = route_add
        self.probability_score[self.steps] = score_add

        self.steps += 1
        self.final_score = self.get_final_score()
        return

    def get_final_score(self):
        scores = numpy.log10(self.probability_score[:self.steps]).sum()

        if self.decoding_alpha > 0.0:
            scores /= self.length_normalization()

        return scores

    def length_normalization(self):
        return ((5 + self.steps) / (5 + 1)) ** self.decoding_alpha

    def copy(self):
        node_clone = SearchNode(max_seq_length=self.max_seq_length,
                                eos_idx=self.eos_idx,
                                decoding_alpha=self.decoding_alpha,
                                device=self.device)
        node_clone.route = self.route.copy()
        node_clone.probability_score = self.probability_score.copy()
        node_clone.steps = self.steps
        node_clone.eos_updated = self.eos_updated
        node_clone.decoding_alpha = self.decoding_alpha

        node_clone.final_score = self.get_final_score()
        return node_clone

    def __lt__(self, other):
        if self.final_score < other.final_score:
            return True
        return False

    def __gt__(self, other):
        if self.final_score > other.final_score:
            return True
        return False

    def __eq__(self, other):
        if self.final_score == other.final_score:
            return True
        return False


class BeamSearch:
    def __init__(self, beam_size: int, max_seq_length: int,
                 tgt_eos_idx: int, device: torch.device,
                 decoding_alpha: float):
        self.beam_size = beam_size
        self.max_seq_length = max_seq_length
        self.device = device
        self.node_pool = []
        self.time_steps = 0
        self.tgt_eos_idx = tgt_eos_idx
        self.decoding_alpha = decoding_alpha

        self.reset()
        return

    def reset(self):
        self.node_pool = [SearchNode(max_seq_length=self.max_seq_length,
                                     eos_idx=self.tgt_eos_idx,
                                     decoding_alpha=self.decoding_alpha,
                                     device=self.device)
                          for _ in range(0, self.beam_size)]
        self.time_steps = 0

    def nodes_to_list(self):
        return sum(list(list(self.node_pool[i].copy() for _ in range(0, self.beam_size)) for i in range(0, self.beam_size)), [])

    def routes(self, probability_scores: torch.Tensor):
        probability_scores = probability_scores.squeeze(dim=1)

        if self.time_steps > 0:
            topk_p, topk_idx = probability_scores.topk(k=self.beam_size, sorted=True)
            topk_p = topk_p.view(-1)
            topk_idx = topk_idx.view(-1)

            pool = self.nodes_to_list()

            for node, p, idx in zip(pool, list(topk_p.split(1, dim=0)), list(topk_idx.split(1, dim=0))):
                node.update(idx, p)

            pool = self.remove_same_nodes(pool)
            scores = torch.Tensor([node.final_score for node in pool]).to(self.device)
            _, sorted_pool_idx = scores.topk(k=self.beam_size, sorted=True)

            pool.sort(reverse=True)
            self.node_pool = pool[:self.beam_size]
        else:
            topk_p, topk_idx = probability_scores[0, :].topk(k=self.beam_size)
            for p, idx, node_temp in zip(topk_p, topk_idx, self.node_pool):
                node_temp.update(idx, p)

        self.time_steps += 1

        return

    def remove_same_nodes(self, pool: [SearchNode]):
        new_pool = []

        for i in range(0, len(pool), self.beam_size):
            if all(node.eos_updated for node in pool[i: i + self.beam_size]):
                new_pool.append(pool[i])
            else:
                new_pool += pool[i: i + self.beam_size]
        return new_pool

    def get_best_route(self):
        best_node = self.node_pool[0]
        return best_node.route[:best_node.steps]

    def all_eos_updated(self):
        if all(node.eos_updated for node in self.node_pool):
            return True
        return False

    def reach_max_length(self):
        if self.time_steps == self.max_seq_length:
            return True
        return False

    def next_input(self):
        return torch.Tensor(list(node.route[:self.time_steps] for node in self.node_pool)).long().to(self.device)
