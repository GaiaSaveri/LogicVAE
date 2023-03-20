#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA

from src.data_utils import *
from src.propkernel import *
from src.proplogic import *


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = np.inf

    def early_stop(self, validation_acc):
        if validation_acc < (self.max_validation_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.max_validation_acc = validation_acc
            self.counter = 0
        return False


def execution_time(start, end, p=False):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    if p:
        print("Execution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return int(hours), int(minutes), int(seconds)


def load_helper(path, model, device, optimizer):
    model.to(device)
    print("\n...Loading ", path, "...")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch, loss, rec_loss, kld_loss = [checkpoint['epoch'], checkpoint['loss'], checkpoint['rec_loss'],
                                       checkpoint['kld_loss']]
    print("...loaded!")
    return {'optimizer': optimizer, 'epoch': epoch, 'loss': loss, 'rec_loss': rec_loss, 'kld_loss': kld_loss}


def load(arg, model, device, optimizer):
    directory = arg.run_folder + os.path.sep + arg.model_name
    filenames = os.listdir(directory)
    file = directory + os.path.sep + sorted(filenames)[-2] if arg.load is None else arg.load
    return load_helper(file, model, device, optimizer)


def save_fn(model, epoch, optimizer, loss_list, arg):
    directory = arg.model_name if arg.run_folder is None else arg.run_folder + os.path.sep + arg.model_name
    filename = arg.model_name + "_epoch="
    filename = filename + str(0)*(6-len(str(epoch))) + str(epoch)
    filename = filename + "_info.pt"
    path = directory + os.path.sep + filename
    os.makedirs(os.path.dirname(directory + "/"), exist_ok=True)
    print("Saving: ", path, "\n")
    assert (len(loss_list) == 3)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list[0], 'rec_loss': loss_list[1], 'kld_loss': loss_list[2]}, path)


def get_semantic_embedding(n_vars, phi_train, phi_test, n_components, device, gram_train=None):
    mu = PropUniformMeasure(varn=n_vars, device=device)
    kernel = PropKernel(mu, samples=None, sigma2=0.44, varn=n_vars)
    if gram_train is None:
        gram_train = kernel.compute_bag_bag(phi_train, phi_train).cpu().numpy()
    gram_test = kernel.compute_bag_bag(phi_test, phi_train).cpu().numpy()
    kpca = PCA(n_components=n_components)
    kpca.fit(gram_train)
    reduced_test = kpca.transform(gram_test)
    reduced_test = torch.from_numpy(reduced_test).to(device)
    if len(phi_test) > 1:
        semantic_init = torch.zeros((len(phi_test) + 2, n_components))
        semantic_init[1:-1] = reduced_test.clone()
        return semantic_init
    return reduced_test
