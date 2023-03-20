#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import torch
from torch.optim.lr_scheduler import CyclicLR

import src.test_dvae
from src.utils import execution_time, save_fn, get_semantic_embedding, EarlyStopper, from_output_to_formula
from src.data_utils import PropFormulaeSymLoader, PropFormulaeLoader, PropFormulaeDataset


def train(arg, model, train_loader, optimizer, enc_phis=None, gram_train=None, save_function=save_fn,
          checkpoint=None):
    start_epoch = 0 if checkpoint is None else checkpoint['epoch'] + 1
    end_epoch = arg.n_epochs
    print("\n...Loading training data...")
    train_batches, stat_name, n_data = train_loader.load_batches(arg, save=True)
    val_batches = train_loader.get_data(kind='validation', simplified=(not arg.var_indexes))
    stats = PropFormulaeDataset.get_data_statistics(PropFormulaeSymLoader.get_plain_dataset(train_batches),
                                                    stat_name, save=True)
    n_components = 0
    if arg.conditional:
        n_components = arg.semantic_length
        train_batches = PropFormulaeSymLoader.get_plain_dataset(train_batches)
        train_phi_list = [from_output_to_formula(d[0][1:-1, 1:-1], d[1][1:-1, 2:]) for d in train_batches]
        val_phi_list = [from_output_to_formula(d[0][1:-1, 1:-1], d[1][1:-1, 2:]) for d in val_batches]
        train_kernel = get_semantic_embedding(arg.n_vars[0], enc_phis, train_phi_list, n_components, model.device,
                                              gram_train=gram_train)
        val_kernel = get_semantic_embedding(arg.n_vars[0], enc_phis, val_phi_list, n_components, model.device,
                                            gram_train=gram_train)
        get_train_kernel = [d.append(train_kernel[d_idx, :].to(model.device)) for d_idx, d in enumerate(train_batches)]
        get_val_kernel = [d.append(val_kernel[d_idx, :].to(model.device)) for d_idx, d in enumerate(val_batches)]
        train_batches = PropFormulaeLoader.divide_batches(train_batches, arg.batch_size, n_data)
    print("...loaded")
    train_loss, rec_loss, kld_loss = [0 for _ in range(3)] if checkpoint is None \
        else [checkpoint['loss'], checkpoint['rec_loss'], checkpoint['kld_loss']]
    # scheduler
    n_batches = len(train_batches)  # number of steps of scheduler per epoch
    n_steps = 2000 - n_batches / 2
    n_steps = n_steps - n_steps % n_batches
    scheduler = CyclicLR(optimizer=optimizer, base_lr=arg.lr, max_lr=3 * arg.lr, mode='triangular',
                         cycle_momentum=False, step_size_up=int(n_steps))
    n_data = None
    save_freq = scheduler.state_dict()['total_size'] / len(train_batches)
    early_stopper = EarlyStopper(patience=3, min_delta=0.03)
    train_start = time.time()
    for epoch in range(start_epoch, end_epoch):
        model.train()
        epoch_loss, epoch_rec_loss, epoch_kld_loss = [0 for _ in range(3)]
        epoch_start = time.time()
        enum_arg = train_batches
        n_data = 0
        for i, g_batch in enumerate(enum_arg):
            model.zero_grad(set_to_none=True)  # optimizer.zero_grad()
            mu, sigma = model.encode(g_batch)
            batch_loss, batch_rec_loss, batch_kld_loss = \
                model.loss(mu, sigma, g_batch, beta=arg.beta)
            batch_loss.backward()
            epoch_loss += batch_loss.item()
            train_loss += float(batch_loss.item())
            epoch_rec_loss += batch_rec_loss.item()
            rec_loss += float(batch_rec_loss.item())
            epoch_kld_loss += batch_kld_loss.item()
            kld_loss += float(batch_kld_loss.item())
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)
            optimizer.step()
            n_data += len(g_batch)
            scheduler.step()
        epoch_end = time.time()
        epoch_h, epoch_m, epoch_s = execution_time(epoch_start, epoch_end)
        div = n_data
        print("Epoch: ", epoch, "Training Total/Reconstruction/KLD Loss: {:.4f}, {:.4f}, {:.4f}".format(
                  epoch_loss / div, epoch_rec_loss / div, epoch_kld_loss / div),
              " [%d, %d, %d]" % (epoch_h, epoch_m, epoch_s))
        file = open(arg.result_folder + os.path.sep + arg.model_name + os.path.sep + "training_results_"
                    + arg.model_name + ".txt", "a")
        file.write("[%d] %f [%d:%d:%d]\n" % (epoch, epoch_loss/div, epoch_h, epoch_m, epoch_s))
        file.close()

        if epoch % save_freq == 0:
            # store checkpoint
            save_function(model, epoch, optimizer, [train_loss, rec_loss, kld_loss], arg)
            model.eval()
            with torch.no_grad():
                val_start = time.time()
                # avg across whole validation set
                val_loss, stats_val = src.test_dvae.test(model, arg, val_batches, kind='validation')
                val_end = time.time()
                val_h, val_m, val_s = execution_time(val_start, val_end)
                print("Epoch: ", epoch, "Validation Total Loss: {:.4f}".format(val_loss),
                      "Validation Accuracy: ", stats_val, " [%d, %d, %d]" % (val_h, val_m, val_s))
                file = open(arg.result_folder + os.path.sep + arg.model_name + os.path.sep + "validation_results_"
                            + arg.model_name + ".txt", "a")
                file.write("[%d] %f %f %f %f %f [%d:%d:%d]\n" % (epoch, val_loss, stats_val[0], stats_val[1],
                                                                 stats_val[2], stats_val[4], val_h, val_m, val_s))
                file.close()
            if early_stopper.early_stop(stats_val[-1]):
                break

    train_end = time.time()
    train_h, train_m, train_s = execution_time(train_start, train_end)
    tot_div = n_data * arg.n_epochs
    print("Training Total/Reconstruction/KLD Loss: {:.4f}, {:.4f}, {:.4f}".format(
        train_loss / tot_div, rec_loss / tot_div, kld_loss / tot_div), " [%d, %d, %d]" % (train_h, train_m, train_s))
