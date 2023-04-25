import os
import math
from decimal import Decimal
from datetime import datetime

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args # configuration augments
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        # handles the training loop for the model
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr() # update the learning rate

        # Get the current time
        current_time = datetime.now()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}\t{}'.format(epoch, Decimal(lr), current_time)
        ) # log the learning rate
        self.loss.start_log()
        self.model.train() # set the model to raining mode

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train): # loop through the batches of the training dataset
            lr, hr = self.prepare(lr, hr) # prepare lr and hr images
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0) # forward process to get super-resolution images
            loss = self.loss(sr, hr) # compute the loess
            loss.backward() # backward propagation to optimize the parameters
            if self.args.gclip > 0: # optionally apply gradient clipping to avoid exploding gradients
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()
            # log the training progress periodically
            if (batch + 1) % self.args.print_every == 0: # every 1000 batches print a learning result

                # Get the current time
                current_time = datetime.now()

                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s\t{}'.format(
                    (batch + 1) * self.args.batch_size, # total number of samples processed by the model upto the current batch
                    len(self.loader_train.dataset), # the length of current train dataset
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(),
                    current_time))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False) # disable gradient computation to save memory

        epoch = self.optimizer.get_last_epoch() # retrieve the last epoch of the optimizer
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval() # set the model to evaluation mode

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background() # if save_results is ture, start to save results in the background
        for idx_data, d in enumerate(self.loader_test): # iterate over the test datasets
            for idx_scale, scale in enumerate(self.scale): # iterate over the scales
                d.dataset.set_scale(idx_scale) # set the current scale
                for lr, hr, filename in tqdm(d, ncols=80): # prepare lr and hr images
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale) # obtain the super resolution image
                    # sr = utility.quantize(sr, self.args.rgb_range) # no need for this step

                    save_list = [sr]
                    # calculate the psnr
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

