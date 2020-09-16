import torch
from torch import nn
import numpy
import time
import os
import re
from torch.nn.functional import pad

from model.model import TransformerMT
from corpus.corpus import Corpus
from optim.optim import set_lr
from utils.stats import Stats
from utils.beam_search import BeamSearch
from utils.bleu import BLEU
from utils.function import chunk, shifted_target, generate_train_mask, generate_loss_mask


class TrainerSingleDevice:
    def __init__(self,
                 model: TransformerMT,
                 corpus: Corpus,
                 optimizer: torch.optim.Optimizer,
                 stats: Stats,
                 bleu: BLEU,
                 tgt_character_level: bool,
                 buffer_every_steps: int,
                 report_every_steps: int,
                 save_every_steps: int,
                 eval_every_steps: int,
                 num_of_steps: int,
                 eval_type: str,
                 processed_steps: int,
                 learning_rate_schedule: str,
                 update_decay: int,
                 batch_capacity: int,
                 max_save_models: int,
                 beam_size: int,
                 decoding_alpha: int,
                 grad_norm_clip: float,
                 grad_norm_clip_type: float,
                 annotate: str,
                 device: torch.device,
                 sample_times: int,
                 exponential_value: int,
                 normalized_cl_factors: bool,
                 ):
        self.model = model
        self.corpus = corpus
        self.optimizer = optimizer
        self.stats = stats
        self.bleu = bleu
        self.tgt_character_level = tgt_character_level

        self.buffer_every_steps = buffer_every_steps
        self.report_every_steps = report_every_steps
        self.save_every_steps = save_every_steps
        self.eval_every_steps = eval_every_steps
        self.num_of_steps = num_of_steps

        self.eval_type = eval_type
        self.processed_steps = processed_steps
        self.update_decay = update_decay
        self.batch_capacity = batch_capacity

        self.src_pad_idx = self.model.src_pad_idx
        self.tgt_eos_idx = self.model.tgt_eos_idx
        self.tgt_pad_idx = self.model.tgt_pad_idx

        self.max_save_models = max_save_models

        self.beam_size = beam_size
        self.decoding_alpha = decoding_alpha

        self.grad_norm_clip = grad_norm_clip if grad_norm_clip > 0.0 else None
        self.grad_norm_clip_type = grad_norm_clip_type

        self.annotate = annotate

        self.device = device

        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.best_bleu = 0.0
        self.best_step = 0

        self.lr_schedule = eval(learning_rate_schedule)
        self.lr = 0.0005
        self.backward_factor = 1.0

        self.loss_report = numpy.zeros(self.report_every_steps, dtype=float)
        self.acc_report = numpy.zeros(self.report_every_steps, dtype=float)
        self.src_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.tgt_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.src_num_pad_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.tgt_num_pad_tokens = numpy.zeros(self.report_every_steps, dtype=int)
        self.num_examples = numpy.zeros(self.report_every_steps, dtype=int)
        self.time_sum = 0.0
        
        self.oom_flag = False
        self.oom_times = 0
        self.oom_recent_records = 0
        self.oom_recent_records_ratio = 0.1
        self.oom_every_steps = 100
        self.update_decay_before_oom = self.update_decay

        self.corpus_segments_thread = self.corpus.invoke_train_segments_making()
        self.corpus_batches_thread = self.corpus.invoke_train_batches_making()

        self.corpus_segments_thread.start()
        self.corpus_batches_thread.start()

        self.sample_times = sample_times
        self.exponential_value = exponential_value
        self.normalized_cl_factors = normalized_cl_factors

        return

    def retrain_model(self,
                      retrain_model: str,
                      processed_steps: int):
        self.processed_steps = processed_steps
        self.corpus.num_of_made_batches.set(processed_steps)
        self.corpus.num_of_trained_batches.set(processed_steps)

        with torch.no_grad():
            print('Loading saved model from %s at step %d ... ' % (retrain_model, processed_steps), end='')
            checkpoint = torch.load(retrain_model)

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            torch.cuda.set_rng_state(checkpoint['gpu_random_state'])

            print('done.')

            print('*' * 80)
            print('Evaluating')
            torch.cuda.empty_cache()
            self.model.eval()
            self.eval_step()
            self.save()
            self.model.train()
            print('Training')

        return

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'gpu_random_state': torch.cuda.get_rng_state()
        }, self.stats.fold_name + '/' + str(self.processed_steps) + '.pt')

        print('Save to:', self.stats.fold_name + '/' + str(self.processed_steps) + '.pt')
        print('*' * 80)

        if self.eval_type == 'acc' and self.processed_steps in self.stats.valid_acc.keys():
            sorted_results = sorted(self.stats.valid_acc.items(), key=lambda d: d[1], reverse=True)
            temp_acc = self.stats.valid_acc[self.processed_steps]
            if self.best_acc < temp_acc:
                print('Best acc: %f -> %f at step %d -> %d' % (self.best_acc, temp_acc,
                                                               self.best_step, self.processed_steps))
                self.best_acc = temp_acc
                self.best_step = self.processed_steps
            else:
                print('Best acc: %f at step %d' % (self.best_acc, self.best_step))

        elif self.eval_type == 'xent' and self.processed_steps in self.stats.valid_loss.keys():
            sorted_results = sorted(self.stats.valid_loss.items(), key=lambda d: d[1])
            temp_loss = self.stats.valid_loss[self.processed_steps]
            if self.best_loss > temp_loss:
                print('Best loss: %f -> %f at step %d -> %d' % (self.best_loss, temp_loss,
                                                                self.best_step, self.processed_steps))
                self.best_loss = temp_loss
                self.best_step = self.processed_steps
            else:
                print('Best loss: %f at step %d' % (self.best_loss, self.best_step))

        elif self.eval_type == 'bleu' and self.processed_steps in self.stats.valid_bleu.keys():
            sorted_results = sorted(self.stats.valid_bleu.items(), key=lambda d: d[1], reverse=True)
            temp_bleu = self.stats.valid_bleu[self.processed_steps]
            if self.best_bleu < temp_bleu:
                print('Best bleu: %f -> %f at step %d -> %d' % (self.best_bleu, temp_bleu,
                                                                self.best_step, self.processed_steps))
                self.best_bleu = temp_bleu
                self.best_step = self.processed_steps
            else:
                print('Best bleu: %f at step %d' % (self.best_bleu, self.best_step))
        else:
            return

        if self.max_save_models > 0 and self.save_every_steps == self.eval_every_steps:
            print('Model performances (%s): ' % self.eval_type)
            for (step_temp, value_temp) in sorted_results[:self.max_save_models]:
                print('%6d\t%8f' % (step_temp, value_temp))

            for (step_temp, _) in sorted_results[self.max_save_models:]:
                path = self.stats.fold_name + '/' + str(step_temp) + '.pt'
                if os.path.isfile(self.stats.fold_name + '/' + str(step_temp) + '.pt'):
                    os.remove(path)
                    print('Remove %d.pt' % step_temp)

        print('*' * 80)

        return

    def run(self):
        while self.processed_steps < self.num_of_steps:
            next_batches = self.corpus.get_train_batches(self.buffer_every_steps)

            for batch in next_batches:
                time_start = time.time()

                while True:
                    if not self.train_step(batch):
                        break
                    else:
                        self.oom_handler()

                self.update()

                self.time_sum += time.time() - time_start

                if self.processed_steps % self.report_every_steps == 0:
                    self.report()

                if self.processed_steps % self.oom_every_steps == 0:
                    print('Last %d steps raise %d OOM errors.' % (self.oom_every_steps, self.oom_recent_records))
                    if self.oom_recent_records / self.oom_every_steps >= self.oom_recent_records_ratio:
                        print('Update decay will be increased from %d to %d.'
                              % (self.update_decay_before_oom, self.update_decay_before_oom + 1))
                        self.update_decay_before_oom += 1
                        self.update_decay = self.update_decay_before_oom
                    self.oom_recent_records = 0

                if self.processed_steps % self.eval_every_steps == 0:
                    with torch.no_grad():
                        print('*' * 80)
                        print('Evaluating')
                        self.model.eval()
                        acc, loss = self.eval_step()
                        bleu_score = self.infer_step()
                        self.stats.valid_record(acc, loss, bleu_score)

                        for i in range(0, self.corpus.num_of_multi_refs):
                            output = str.format('Step %6d valid, ref%1d acc: %5.2f loss: %5.2f bleu: %f'
                                                % (self.processed_steps, i, acc[i] * 100, loss[i], bleu_score))
                            print(output)
                            self.stats.log_to_file(output)

                        print('*' * 80)
                        torch.cuda.empty_cache()
                        self.model.train()
                        print('Training')
                        print(self.annotate)

                if self.processed_steps % self.save_every_steps == 0:
                    self.save()

                if self.processed_steps >= self.num_of_steps:
                    print('End of training.')
                    return
        return

    def report(self):
        for acc_step, loss_step in zip(self.acc_report.tolist(), self.loss_report.tolist()):
            self.stats.train_record(acc_step, loss_step)

        output_str = str.format(
            'Step: %6d, acc:%6.2f (%6.2f~%6.2f), loss:%5.2f (%5.2f~%5.2f), '
            'lr: %.4f, bc: %d/%d, bs: %5d, tks: %6d+%6d, t: %5.2f'
            % (self.processed_steps,
               self.acc_report.mean() * 100,
               self.acc_report.min() * 100,
               self.acc_report.max() * 100,
               self.loss_report.mean(),
               self.loss_report.min(),
               self.loss_report.max(),
               self.lr,
               self.src_tokens.sum() + self.tgt_tokens.sum(),
               self.src_num_pad_tokens.sum() + self.tgt_num_pad_tokens.sum(),
               self.num_examples.sum(),
               self.src_tokens.sum(),
               self.tgt_tokens.sum(),
               self.time_sum
               )
        )

        self.stats.log_to_file(output_str)
        print(output_str)

        self.acc_report.fill(0)
        self.loss_report.fill(0)
        self.src_tokens.fill(0)
        self.tgt_tokens.fill(0)
        self.src_num_pad_tokens.fill(0)
        self.tgt_num_pad_tokens.fill(0)
        self.num_examples.fill(0)
        self.time_sum = 0.0

        return

    def update(self):
        if self.grad_norm_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.grad_norm_clip,
                                     norm_type=self.grad_norm_clip_type)

        self.processed_steps += 1
        self.lr = self.lr_schedule(self.processed_steps)
        set_lr(self.optimizer, self.lr)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.embeddings.zero_pad_emb()

        return

    def train_step(self, batch):
        report_idx = self.processed_steps % self.report_every_steps
        src, tgt, src_lens, tgt_lens = batch

        self.src_num_pad_tokens[report_idx] = int(src.numel())
        self.tgt_num_pad_tokens[report_idx] = int(tgt.numel())
        self.src_tokens[report_idx] = int(src_lens.sum())
        self.tgt_tokens[report_idx] = int(tgt_lens.sum())
        self.num_examples[report_idx] = int(src.size(0))

        self.oom_flag = False

        tgt_in, tgt_out = shifted_target(target_tensor=tgt, tgt_eos_idx=self.tgt_eos_idx, tgt_pad_idx=self.tgt_pad_idx)
        src_mask, tgt_mask, crs_mask = generate_train_mask(mask_backup=self.model.mask_backup,
                                                           triu_backup=self.model.triu_backup,
                                                           source_lengths=src_lens,
                                                           target_lengths=tgt_lens,
                                                           batch_size=self.num_examples[report_idx],
                                                           src_len_max=src.size(1),
                                                           tgt_len_max=tgt_in.size(1))
        sampled_cl_factor_token, sampled_cl_factor_sent = \
            _monte_carlo_sample(model=self.model,
                                src_input=src,
                                tgt_input=tgt_in,
                                tgt_output=tgt_out,
                                src_attn_mask=src_mask,
                                tgt_attn_mask=tgt_mask,
                                crs_attn_mask=crs_mask,
                                tgt_lens=tgt_lens - 1,
                                sample_times=self.sample_times,
                                exponential_value=self.exponential_value
                                )
        sampled_cl_factor_mask = tgt[:, 1:].eq(self.tgt_pad_idx).cuda()
        sampled_cl_factor = sampled_cl_factor_token * sampled_cl_factor_sent.unsqueeze(dim=-1)

        if self.normalized_cl_factors:
            sampled_cl_factor.masked_fill_(mask=sampled_cl_factor_mask, value=float('-inf'))
            sampled_cl_factor = torch.softmax(sampled_cl_factor.view(-1), dim=-1).view_as(sampled_cl_factor_mask) * int((tgt_lens - 1).sum())
        else:
            sampled_cl_factor.masked_fill_(mask=sampled_cl_factor_mask, value=0.0)

        sampled_factors_nopad = sampled_cl_factor.masked_select(mask=~sampled_cl_factor_mask)
        print("Factor: %.4e ~ %.4e" % (float(sampled_factors_nopad.min()), float(sampled_factors_nopad.max())))

        try:
            all_acc_tokens = 0.0
            all_loss = 0.0

            for s, t, sl, tl, cl_factor in chunk(src, tgt, src_lens, tgt_lens, sampled_cl_factor, self.update_decay):
                t_in, t_out = shifted_target(t, self.tgt_eos_idx, self.tgt_pad_idx)
                bs, sl_max = s.size()
                tl_max = int(t_in.size(1))
                sm, tm, cm = generate_train_mask(self.model.mask_backup, self.model.triu_backup, sl, tl - 1, bs, sl_max, tl_max)
                output, loss = self.model(s, t_in, sm, tm, cm, t_out)
                loss_mask = generate_loss_mask(self.model.mask_backup, bs, tl - 1, tl_max, 1.0 / self.num_examples[report_idx], cl_factor)
                loss.backward(loss_mask)

                loss_value = float(loss.sum()) / (float(tl.sum()) - bs) * bs
                tgt_mask = t_out.ne(self.tgt_pad_idx)
                acc_value = float(output.masked_select(tgt_mask).eq(t_out.masked_select(tgt_mask)).sum().float())

                all_acc_tokens += acc_value
                all_loss += loss_value

        except RuntimeError as e:
            print(e.args[0])
            self.oom_flag = True
            # raise e

        if self.oom_flag:
            self.oom_times += 1
            print('OOM Error raises. Try to increase update decay from %d to %d and run over the same batch again.'
                  % (self.update_decay, self.update_decay + 1))
            if self.oom_times > 5:
                print('OOM Error in 5 contiguous times. Update decay will be permanently increased 1 from %d to %d.'
                      % (self.update_decay, self.update_decay + 1))
                self.update_decay_before_oom += 1
                self.oom_recent_records = 0
            self.update_decay += 1
            return True
        else:
            if self.update_decay == self.update_decay_before_oom:
                self.oom_times = 0
            else:
                self.oom_recent_records += 1
            self.update_decay = self.update_decay_before_oom

        self.acc_report[report_idx] += all_acc_tokens / self.tgt_tokens[report_idx]
        self.loss_report[report_idx] += all_loss / self.num_examples[report_idx]

        return

    def eval_step(self):
        acc = numpy.zeros(self.corpus.num_of_multi_refs)
        loss = numpy.zeros(self.corpus.num_of_multi_refs)

        source_data, target_data, src_lens, tgt_lens = self.corpus.get_valid_batches()
        num_of_examples = len(self.corpus.corpus_source_valid_numerate)

        for s, sl, t, tl in zip(source_data, src_lens, target_data, tgt_lens):
            for i in range(0, self.corpus.num_of_multi_refs):
                t_sub = t[i]
                tl_sub = tl[i]
                t_in, t_out = shifted_target(t_sub, self.tgt_eos_idx, self.tgt_pad_idx)
                bs, sl_max = s.size()
                tl_max = int(t_in.size(1))
                sm, tm, cm = generate_train_mask(self.model.mask_backup, self.model.triu_backup, sl, tl_sub, bs, sl_max, tl_max)
                output, loss = self.model(s, t_in, sm, tm, cm, t_out)

                loss_value = float(loss.sum()) / (float(tl_sub.sum()) - bs) * bs
                tgt_mask = t_out.ne(self.tgt_pad_idx)
                acc_value = float(output.masked_select(tgt_mask).eq(t_out.masked_select(tgt_mask)).sum().float()) \
                            / float(tl_sub.sum() - bs) * bs

                acc[i] += acc_value
                loss[i] += loss_value

        acc /= num_of_examples
        loss /= num_of_examples

        return acc, loss

    def infer_step(self):
        src, tgt = self.corpus.get_valid_batches_for_translation()
        num_of_batches = len(src)

        translation_results = list()
        for idx, s in enumerate(src):
            print('\rTranslating batch %d/%d ... ' % (idx + 1, num_of_batches), sep=' ', end='')

            batch_size, sl_max = s.size()
            device = self.model.triu_backup.device
            tgt_sos_token = torch.Tensor([self.model.tgt_sos_idx]).view((1, 1)) \
                .repeat((batch_size * self.beam_size, 1)).contiguous().long().to(device)

            if self.model.infer_max_seq_length_mode == 'absolute':
                tgt_infer_max_seq_length = list(self.model.infer_max_seq_length for _ in range(0, batch_size))
            else:
                sl = s.ne(self.src_pad_idx).sum(dim=-1).cpu().tolist()
                tgt_infer_max_seq_length = list(x + self.model.infer_max_seq_length for x in sl)
            beam_search = list(BeamSearch(beam_size=self.beam_size,
                                          max_seq_length=l,
                                          tgt_eos_idx=self.tgt_eos_idx,
                                          device=device,
                                          decoding_alpha=self.decoding_alpha) for l in tgt_infer_max_seq_length)

            src_mask = s.eq(self.src_pad_idx).unsqueeze(dim=1).expand(size=(batch_size, sl_max, sl_max))
            crs_mask_temp = s.eq(self.src_pad_idx).unsqueeze(dim=1)

            src_embs = self.model.embeddings.get_src_embs(s)
            context_vector = self.model.encoder(src_embs, src_mask)
            next_input = tgt_sos_token
            non_updated_index = list(range(0, batch_size))
            non_updated_index_tensor = torch.Tensor(non_updated_index).long().to(device)

            context_vector_step = torch.cat(list(x.unsqueeze(dim=0).repeat(self.beam_size, 1, 1)
                                                 for x in context_vector.unbind(dim=0)), dim=0)

            for i in range(0, self.model.infer_max_seq_length if self.model.infer_max_seq_length_mode == 'absolute'
            else self.model.infer_max_seq_length + sl_max):
                if batch_size > non_updated_index_tensor.numel():
                    context_vector_step = context_vector.index_select(dim=0, index=non_updated_index_tensor)
                    context_vector_step = torch.cat(list(x.unsqueeze(dim=0).repeat(self.beam_size, 1, 1)
                                                         for x in context_vector_step.unbind(dim=0)), dim=0)
                tgt_mask = self.model.triu_backup[:i + 1, :i + 1].unsqueeze(dim=0).expand(
                    (len(non_updated_index) * self.beam_size, i + 1, i + 1))
                crs_mask = crs_mask_temp.index_select(dim=0, index=non_updated_index_tensor).repeat(repeats=(1, i + 1, 1))
                crs_mask = torch.cat(list(x.unsqueeze(dim=0).repeat(self.beam_size, 1, 1) for x in crs_mask.unbind(dim=0)), dim=0)
                tgt_embs = self.model.embeddings.get_tgt_embs(next_input)

                step_output = self.model.decoder(tgt_embs, tgt_mask, context_vector_step, crs_mask)
                step_output_logits = self.model.embeddings.get_logits(step_output[:, -1:, :])
                step_output_softmax = self.model.softmax(step_output_logits, dim=-1)
                step_output_softmax = pad(step_output_softmax, pad=[1, 0], mode='constant', value=0.0)

                for idx, probs in zip(non_updated_index, step_output_softmax.split(self.beam_size, dim=0)):
                    beam_search[idx].routes(probs)

                non_updated_index = list(x[0] for x in filter(
                    lambda d: not (d[1].all_eos_updated() or d[1].reach_max_length()), enumerate(beam_search)))
                non_updated_index_tensor = torch.Tensor(non_updated_index).long().to(device)

                if len(non_updated_index) == 0:
                    break

                next_input = torch.cat((tgt_sos_token[:len(non_updated_index) * self.beam_size],
                                        torch.cat(list(beam_search[idx].next_input() for idx in non_updated_index), dim=0)),
                                       dim=-1)

            result = list(beam.get_best_route().tolist() for beam in beam_search)
            translation_results += result

        print('done.')

        bleu_score = self.bleu.bleu(translation_results, tgt)
        print('BLEU score: %5.2f' % (bleu_score * 100))

        if self.tgt_character_level:
            r = re.compile(r'((?:(?:[a-zA-Z0-9])+[\-\+\=!@#\$%\^&\*\(\);\:\'\"\[\]{},\.<>\/\?\|`~]*)+|[^a-zA-Z0-9])')
            print('')
            print('For character-level:')

            hyp_char = list(' '.join(sum(list(r.findall(x) for x in line), list())).split() for line in translation_results)
            ref_char = list(list(' '.join(sum(list(r.findall(x) for x in line), list())).split()
                                                for line in gt_ref) for gt_ref in tgt)
            bleu_score = self.bleu.bleu(hyp_char, ref_char)
            print('BLEU score: %5.2f' % (bleu_score * 100))

        return bleu_score

    def release(self):
        print('Releasing required resources ... ', end='')
        self.corpus.running.set(False)
        time.sleep(5)
        self.corpus_segments_thread.terminate()
        self.corpus_batches_thread.terminate()
        time.sleep(5)
        self.corpus_segments_thread.close()
        self.corpus_batches_thread.close()
        print('done.')
        return

    def oom_handler(self):
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        return


def _monte_carlo_sample(model: TransformerMT,
                        src_input: torch.Tensor,
                        tgt_input: torch.Tensor,
                        tgt_output: torch.Tensor,
                        src_attn_mask: torch.Tensor,
                        tgt_attn_mask: torch.Tensor,
                        crs_attn_mask: torch.Tensor,
                        tgt_lens: torch.Tensor,
                        sample_times: int,
                        exponential_value: int):
    all_probs = list()
    
    with torch.no_grad():
        for _ in range(0, sample_times):
            src_embs = model.embeddings.get_src_embs(src_input)
            context_vector = model.encoder(src_embs, src_attn_mask)
            tgt_embs = model.embeddings.get_tgt_embs(tgt_input)
            output = model.decoder(tgt_embs, tgt_attn_mask, context_vector, crs_attn_mask)
            logits = model.embeddings.get_logits(output).softmax(dim=-1)
            logits = pad(logits.view(-1, logits.size(-1)), pad=[1, 0])
            new_idxs = torch.arange(start=0, end=logits.size(0), dtype=torch.long, device=logits.device)\
                        * logits.size(1)
            probs = logits.take(new_idxs + tgt_output.view(-1))
            all_probs.append(probs)

    # all_probs = torch.cat(list(x.unsqueeze_(dim=0) for x in all_probs), dim=0)
    # token_var = all_probs.var(dim=0).view(src_input.size(0), -1)

    # token_loss_mask = (1 - token_var) ** exponential_value
    
    # all_probs.log_()
    # pad_mask = model.mask_backup[:, :tgt_input.size(-1)].repeat(tgt_input.size(0), 1).ge(tgt_lens.unsqueeze(dim=-1)).view(-1)
    # all_probs.masked_fill_(mask=pad_mask, value=0.0)
    # all_probs = all_probs.view(sample_times, src_input.size(0), -1)
    
    # all_probs = all_probs.sum(dim=-1)
    # all_probs_min = all_probs.min(dim=-1)[0]

    # all_probs = all_probs / all_probs_min.unsqueeze_(dim=-1)
    # sent_var = all_probs.var(dim=0)

    # sent_loss_mask = (1 - sent_var) ** exponential_value

    all_probs = torch.cat(list(x.unsqueeze_(dim=0) for x in all_probs), dim=0)
    all_probs.log_()
    pad_mask = model.mask_backup[:, :tgt_input.size(-1)].repeat(tgt_input.size(0), 1).ge(tgt_lens.unsqueeze(dim=-1)).view(-1)
    all_probs.masked_fill_(mask=pad_mask, value=0.0)

    all_probs_token_min = all_probs.min(dim=-1)[0]
    all_probs_token = all_probs / all_probs_token_min.unsqueeze(dim=-1)
    token_var = all_probs_token.var(dim=0).view(src_input.size(0), -1)

    token_loss_mask = (1 - token_var) ** exponential_value
    
    all_probs = all_probs.view(sample_times, src_input.size(0), -1)
    
    all_probs = all_probs.sum(dim=-1)
    all_probs_min = all_probs.min(dim=-1)[0]

    all_probs = all_probs / all_probs_min.unsqueeze_(dim=-1)
    sent_var = all_probs.var(dim=0)

    sent_loss_mask = (1 - sent_var) ** exponential_value
    
    return token_loss_mask, sent_loss_mask
