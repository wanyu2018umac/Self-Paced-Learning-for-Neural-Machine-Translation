import torch
from torch.nn.functional import pad
from copy import deepcopy
import re

from corpus.corpus import Corpus
from model.model import TransformerMT
from utils.beam_search import BeamSearch
from utils.bleu import BLEU


class TranslatorEnsemble:
    def __init__(self,
                 corpus: Corpus,
                 bleu: BLEU,
                 model: TransformerMT,
                 model_paths: [str],
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 tgt_sos_idx: int,
                 tgt_eos_idx: int,
                 tgt_character_level,
                 beam_size: [int],
                 decoding_alpha: [float],
                 output_prefix: str,
                 output_suffix: str,
                 save_output: bool,
                 src_vocab_paths: [str],
                 tgt_vocab_paths: [str]):
        self.corpus = corpus
        self.bleu = bleu
        self.model = model
        self.model_paths = model_paths

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx

        self.tgt_character_level = tgt_character_level

        self.beam_size = beam_size
        self.decoding_alpha = decoding_alpha

        self.save_output = save_output
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix

        self.src_vocab_paths = src_vocab_paths
        self.tgt_vocab_paths = tgt_vocab_paths

        self.model_list = list()

        return

    def model_ensemble(self):
        print('Loading model from %s ... ' % self.model_paths[0], end='')
        checkpoint = torch.load(self.model_paths[0])
        self.model.load_state_dict(checkpoint['model'])
        print('done.')
        self.model_list.append(self.model)
        device = next(self.model.parameters()).device

        for model_path in self.model_paths[1:]:
            model_temp = deepcopy(self.model).to(device)
            print('Loading model from %s ... ' % model_path, end='')
            checkpoint = torch.load(model_path)
            model_temp.load_state_dict(checkpoint['model'])
            print('done.')
            self.model_list.append(model_temp)

        print('Ensembling model ... ', end='')
        self.gather_embeddings()
        print('done.')
        print('*' * 80)

        return

    def gather_embeddings(self):
        src_vocab_origin = torch.load(self.src_vocab_paths[0])
        src_vocab_origin = dict(sorted(src_vocab_origin.items(), key=lambda d: d[1]))
        tgt_vocab_origin = torch.load(self.tgt_vocab_paths[0])
        tgt_vocab_origin = dict(sorted(tgt_vocab_origin.items(), key=lambda d: d[1]))
        device = next(self.model.parameters()).device

        for src_vocab_path, tgt_vocab_path, model in \
                zip(self.src_vocab_paths[1:], self.tgt_vocab_paths[1:], self.model_list[1:]):
            src_vocab_sub = torch.load(src_vocab_path)
            tgt_vocab_sub = torch.load(tgt_vocab_path)

            src_idxs = torch.Tensor(list(src_vocab_sub[word] for word in src_vocab_origin)).long().to(device)
            tgt_idxs = torch.Tensor(list(tgt_vocab_sub[word] for word in tgt_vocab_origin)).long().to(device)

            if model.share_embedding and model.share_projection_and_embedding:
                model.embeddings.embs.data = model.embeddings.data.index_select(dim=0, index=src_idxs)
            elif model.share_embedding:
                model.embeddings.embs.data = model.embeddings.data.index_select(dim=0, index=src_idxs)
                model.embeddings.logits.data = model.embeddings.logits.data.index_select(dim=0, index=tgt_idxs)
            elif model.share_projection_and_embedding:
                model.embeddings.src_embs.data = model.embeddings.src_embs.data.index_select(dim=0, index=src_idxs)
                model.embeddings.tgt_embs.data = model.embeddings.tgt_embs.data.index_select(dim=0, index=tgt_idxs)
            else:
                model.embeddings.src_embs.data = model.embeddings.src_embs.data.index_select(dim=0, index=src_idxs)
                model.embeddings.tgt_embs.data = model.embeddings.tgt_embs.data.index_select(dim=0, index=tgt_idxs)
                model.embeddings.logits.data = model.embeddings.logits.data.index_select(dim=0, index=tgt_idxs)

        return

    def translate(self):
        print('Testing ensembled model from models: \n\t%s' % '\n\t'.join(self.model_paths))
        print('*' * 80)

        self.model_ensemble()

        for num_of_test in range(0, len(self.corpus.corpus_source_test_name)):
            result_matrix = dict()
            result_matrix_char = dict()

            for beam_size in self.beam_size:
                for decoding_alpha in self.decoding_alpha:
                    if beam_size == 1 and self.decoding_alpha.index(decoding_alpha) > 0:
                        print('Beam size equal to 1 means greedy decoding, so skip all other decoding_alpha values.')
                        continue

                    with torch.no_grad():
                        bleu_score, hyp, bleu_score_char, hyp_char = \
                            self.infer_step(beam_size=beam_size, decoding_alpha=decoding_alpha, num_of_test=num_of_test)

                    if beam_size not in result_matrix.keys():
                        result_matrix[beam_size] = list()
                    result_matrix[beam_size].append(bleu_score)

                    if beam_size not in result_matrix_char.keys():
                        result_matrix_char[beam_size] = list()
                    result_matrix_char[beam_size].append(bleu_score_char)

                    hyp_tofile = hyp_char if self.tgt_character_level else hyp
                    hyp_filename = '%s-esb-%d-%.2f-%d-%s' % \
                                   (self.output_prefix, beam_size, decoding_alpha, num_of_test, self.output_suffix)

                    if self.save_output:
                        with open(hyp_filename, mode='w', encoding='utf-8') as f:
                            for hyp_line in hyp_tofile:
                                f.write(' '.join(hyp_line) + '\n')

                    print('*' * 80)

            print('Performance matrix:')
            print('Horizontal: decoding alpha; Vertical: beam size')
            print('\t' + '\t'.join('%5.2f' % x for x in self.decoding_alpha))
            for beam_size in result_matrix.keys():
                print('%2d\t' % beam_size + '\t'.join('%7.4f' % (x * 100) for x in result_matrix[beam_size]))

            if self.tgt_character_level:
                print('*' * 80)
                print('Character level:')
                print('\t' + '\t'.join('%5.2f' % x for x in self.decoding_alpha))
                for beam_size in result_matrix_char.keys():
                    print('%2d\t' % beam_size + '\t'.join('%7.4f' % (x * 100) for x in result_matrix_char[beam_size]))

            print('*' * 80)
            print('*' * 80)
        return

    def infer_step(self, beam_size: int, decoding_alpha: float, num_of_test: int):
        src, tgt, order = self.corpus.get_test_batches(num_of_test)
        num_of_batches = len(src)

        hyp = list()
        for idx, s in enumerate(src):
            print('\rTranslating batch %d/%d ... ' % (idx + 1, num_of_batches), sep=' ', end='')

            batch_size, sl_max = s.size()
            device = self.model.triu_backup.device
            tgt_sos_token = torch.Tensor([self.tgt_sos_idx]).view((1, 1)) \
                .repeat((batch_size * beam_size, 1)).contiguous().long().to(device)

            if self.model.infer_max_seq_length_mode == 'absolute':
                tgt_infer_max_seq_length = list(self.model.infer_max_seq_length for _ in range(0, batch_size))
            else:
                sl = s.ne(self.src_pad_idx).sum(dim=-1).cpu().tolist()
                tgt_infer_max_seq_length = list(x + self.model.infer_max_seq_length for x in sl)
            beam_search = list(BeamSearch(beam_size=beam_size,
                                          max_seq_length=l,
                                          tgt_eos_idx=self.tgt_eos_idx,
                                          device=device,
                                          decoding_alpha=decoding_alpha) for l in tgt_infer_max_seq_length)

            src_mask = s.eq(self.src_pad_idx).unsqueeze(dim=1).expand(size=(batch_size, sl_max, sl_max))
            crs_mask_temp = s.eq(self.src_pad_idx).unsqueeze(dim=1)

            context_vector_list = list()
            for model in self.model_list:
                src_embs = model.embeddings.get_src_embs(s)
                context_vector = model.encoder(src_embs, src_mask)
                context_vector_list.append(context_vector)

            next_input = tgt_sos_token
            non_updated_index = list(range(0, batch_size))
            non_updated_index_tensor = torch.Tensor(non_updated_index).long().to(device)

            for i in range(0, self.model.infer_max_seq_length if self.model.infer_max_seq_length_mode == 'absolute' \
                    else self.model.infer_max_seq_length + sl_max):
                tgt_mask = self.model.triu_backup[:i + 1, :i + 1].unsqueeze(dim=0).expand(
                    (len(non_updated_index) * beam_size, i + 1, i + 1))
                crs_mask = crs_mask_temp.index_select(dim=0, index=non_updated_index_tensor).repeat(
                    repeats=(1, i + 1, 1))
                crs_mask = torch.cat(list(x.unsqueeze(dim=0).repeat(beam_size, 1, 1) for x in crs_mask.unbind(dim=0)),
                                     dim=0)

                step_output_softmax_list = list()
                for context_vector, model in zip(context_vector_list, self.model_list):
                    context_vector_step = torch.cat(list(x.unsqueeze(dim=0).repeat(beam_size, 1, 1)
                                                         for x in context_vector.unbind(dim=0)), dim=0)
                    if batch_size > non_updated_index_tensor.numel():
                        context_vector_step = context_vector_step.index_select(dim=0, index=non_updated_index_tensor)
                        context_vector_step = torch.cat(list(x.unsqueeze(dim=0).repeat(beam_size, 1, 1)
                                                             for x in context_vector_step.unbind(dim=0)), dim=0)

                    tgt_embs = self.model.embeddings.get_tgt_embs(next_input)

                    step_output = self.model.decoder(tgt_embs, tgt_mask, context_vector_step, crs_mask)
                    step_output_logits = self.model.embeddings.get_logits(step_output[:, -1:, :])
                    step_output_softmax = self.model.softmax(step_output_logits, dim=-1)
                    step_output_softmax = pad(step_output_softmax, pad=[1, 0], mode='constant', value=0.0)
                    step_output_softmax_list.append(step_output_softmax)

                step_output_softmax_averaged = sum(step_output_softmax_list) / len(self.model_list)
                for idx, probs in zip(non_updated_index, step_output_softmax_averaged.split(beam_size, dim=0)):
                    beam_search[idx].routes(probs)

                non_updated_index = list(x[0] for x in filter(
                    lambda d: not (d[1].all_eos_updated() or d[1].reach_max_length()), enumerate(beam_search)))
                non_updated_index_tensor = torch.Tensor(non_updated_index).long().to(device)

                if len(non_updated_index) == 0:
                    break

                next_input = torch.cat((tgt_sos_token[:len(non_updated_index) * beam_size],
                                        torch.cat(list(beam_search[idx].next_input() for idx in non_updated_index), dim=0)),
                                       dim=-1)

            result = list(beam.get_best_route().tolist() for beam in beam_search)
            hyp += result

        print('done.')

        if self.corpus.bpe_tgt:
            hyp = list(self.corpus.byte_pair_handler_tgt.subwords2words(
                list(self.corpus.tgt_idx2word[x] for x in l)) for l in hyp)
            tgt = list(list(self.corpus.byte_pair_handler_tgt.subwords2words(
                list(self.corpus.tgt_idx2word[x] for x in l)) for l in ref) for ref in tgt)
        else:
            hyp = list(list(self.corpus.tgt_idx2word[x] for x in l) for l in hyp)
            tgt = list(list(list(self.corpus.tgt_idx2word[x] for x in l) for l in ref) for ref in tgt)

        bleu_score = self.bleu.bleu(hyp, tgt)
        print('BLEU score: %5.2f' % (bleu_score * 100))

        if self.tgt_character_level:
            r = re.compile(r'((?:(?:[a-zA-Z0-9])+[\-\+\=!@#\$%\^&\*\(\);\:\'\"\[\]{},\.<>\/\?\|`~]*)+|[^a-zA-Z0-9])')
            print('')
            print('For character-level:')

            hyp_char = list(' '.join(sum(list(r.findall(x) for x in line), list())).split() for line in hyp)
            tgt_char = list(list(' '.join(sum(list(r.findall(x) for x in line), list())).split()
                                 for line in gt_ref) for gt_ref in tgt)

            bleu_score_char = self.bleu.bleu(hyp_char, tgt_char)
            print('BLEU score: %5.2f' % (bleu_score * 100))

            hyp_char = list(x[1] for x in sorted(zip(order, hyp_char), key=lambda d: d[0]))
        else:
            bleu_score_char = 0.0
            hyp_char = None

        hyp = list(x[1] for x in sorted(zip(order, hyp), key=lambda d: d[0]))

        return bleu_score, hyp, bleu_score_char, hyp_char
