import torch
import time
import numpy
import os


class Stats:
    def __init__(self, num_of_steps: int, processed_steps: int):
        self.train_acc = torch.zeros(num_of_steps)
        self.train_loss = torch.zeros(num_of_steps)

        self.valid_acc = {}
        self.valid_loss = {}
        self.valid_bleu = {}

        self.test_bleu = {}

        self.time = time.localtime(time.time())
        self.fold_name = '{0:0>4d}{1:0>2d}{2:0>2d}{3:0>2d}{4:0>2d}{5:0>2d}'\
            .format(self.time.tm_year, self.time.tm_mon, self.time.tm_mday,
                    self.time.tm_hour, self.time.tm_min, self.time.tm_sec)

        if os.path.isdir(self.fold_name):
            print('Fold exists.')
        else:
            os.mkdir(self.fold_name)

        self.file_name = self.fold_name + '/log.txt'
        self.train_time_step = processed_steps
        print('Log file:', self.file_name)
        return

    def train_record(self, acc: float, loss: float):
        self.train_acc[self.train_time_step] = acc
        self.train_loss[self.train_time_step] = loss
        self.train_time_step += 1
        return

    def valid_record(self, acc: numpy.array, loss: numpy.array, bleu: numpy.array):
        self.valid_acc[self.train_time_step] = acc
        self.valid_loss[self.train_time_step] = loss
        self.valid_bleu[self.train_time_step] = bleu
        return

    def log_to_file(self, string: str):
        t = time.localtime(time.time())
        prefix = '{0:0>4d}{1:0>2d}{2:0>2d}{3:0>2d}{4:0>2d}{5:0>2d}' \
            .format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        with open(self.file_name, 'a') as f:
            f.write(prefix + '\t' + string + '\n')
        return

    def valid_stats_to_file(self):
        self.log_to_file('valid\tref no.\tacc\tloss\tbleu')
        with open(self.file_name, 'a') as f:
            for key in sorted(self.valid_acc.keys()):
                for i in range(0, len(self.valid_acc[key])):
                    f.write('{0:d}\t{1:d}\t{2:5.2f}\t{3:5.2f}\t{4:5.2f}\n'.
                            format(key, i, self.valid_acc[key][i] * 100, self.valid_loss[key][i], self.valid_bleu[key] * 100))
        self.log_to_file('*' * 80)
        return

    def train_stats_to_file(self):
        self.log_to_file('train\tacc\tloss')
        with open(self.file_name, 'a') as f:
            for i in range(0, self.train_time_step):
                f.write('{0:d}\t{1:5.2f}\t{2:5.2f}\n'.format(i, self.train_acc[i].item() * 100, self.train_loss[i].item()))
        self.log_to_file('*' * 80)
        return

    def get_best_acc(self):
        return max(v.mean() for v in self.valid_acc.values())

    def get_best_loss(self):
        return min(v.mean() for v in self.valid_loss.values())

    def get_best_bleu(self):
        return max(self.valid_bleu.values())
