

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import operator
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from numbers import Number
from shutil import copyfile

import numpy as np
import torch
import tqdm


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.3f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device, name=None, domain=None):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x, domain=domain)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total * 100

def plot_tsne(feats_list, labels_list, step_list, filename, mode):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D 
    
    # Define markers for each step
    markers = ['o', '+', '*']  # Extend this list if needed
    
    # Define colors for each label - assuming 7 labels
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Red, Green, Blue, Cyan, Magenta, Yellow, Black
    
    # Concatenate all features and labels for a combined t-SNE transformation
    all_feats = np.concatenate([feats.cpu().numpy() for feats in feats_list], axis=0)
    all_labels = np.concatenate([labels.cpu().numpy() for labels in labels_list], axis=0)
    
    # Perform t-SNE on the combined dataset
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(all_feats)
    
    plt.figure(figsize=(16,10))
    # Initialize starting index for slicing tsne_results
    start_idx = 0
    for i, (feats, labels, step) in enumerate(zip(feats_list, labels_list, step_list)):
        num_points = feats.shape[0]  # Number of points in the current set
        # Slice the t-SNE results for the current set
        current_tsne_results = tsne_results[start_idx:start_idx + num_points]
        start_idx += num_points  # Update the start index for the next set
        
        if (mode == 'only_trg' and i == 0) or (mode == 'src_trg_before' and i == 2) or (mode == 'src_trg_after'  and i == 1):
            continue
        
        # Use different markers for each step
        marker = markers[i]
        
        # Plot
        for label in np.unique(all_labels):
            idx = (labels.cpu().numpy() == label)
            plt.scatter(current_tsne_results[idx, 0], current_tsne_results[idx, 1], color=colors[label], marker=marker, alpha=0.7, s=60)
    
    # Custom legends for markers (steps) and colors (labels)
    legend_elements_marker = [Line2D([0], [0], marker=marker, color='k', label=step, markersize=10, markerfacecolor='k') for marker, step in zip(markers, step_list)]
    legend_elements_color = [Line2D([0], [0], marker='o', color=color, label=f'Label {i}', markersize=10) for i, color in enumerate(colors)]
    
    # Add custom legends
    legend1 = plt.legend(handles=legend_elements_marker, title="Steps", loc="upper left")
    
    plt.gca().add_artist(legend1)  # Add the first legend explicitly
    plt.legend(handles=legend_elements_color, title="Labels", loc="lower left")
    plt.title(mode)
    plt.savefig(filename)
    plt.close()

    

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
    
# extract feature without prompt
def forward_feature(network, loader, device):
    features = []
    labels = []
    logit_list = []
    
    correct = 0
    total = 0
    
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.featurizer(x)
            features.append(p.detach().cpu())
            labels.append(y.detach().cpu())
            
            logit = network.classifier(p)
            total += y.size(0)
            correct += (logit.argmax(1) == y).sum().item()
            logit_list.append(logit.detach().cpu())
            
    features = torch.cat(features)
    labels = torch.cat(labels)
    # logit_list = torch.cat(logit_list)
    
    print(f'Accuracy: {correct / total*100:.3f}')
    
    return features, labels
    
# load source features if exists, otherwise compute and save
def load_src_features(args, tta, src_test_loaders, device, sample_per_class=10):
    src_features, src_labels = [], []
    if os.path.exists(os.path.join(args.output_dir, 'imagenet_val_features_labels.pkl')):
        # print(f'Loading from pre-computed source features')
        src_features_labels = torch.load(os.path.join(args.output_dir, 'imagenet_val_features_labels.pkl'))
        src_features = src_features_labels['src_features']
        src_labels = src_features_labels['src_labels']
    else:
        for i, loader in enumerate(src_test_loaders):
            features,  labels = forward_feature(tta, loader, device)
            src_features.append(features)
            src_labels.append(labels)
        
        src_features = torch.cat(src_features, dim=0)
        src_labels = torch.cat(src_labels, dim=0)
        
        # print(f'src_features: {src_features.shape}, src_labels: {src_labels.shape}, saved to {os.path.join(args.output_dir, "imagenet_val_features_labels.pkl")}')
        torch.save({'src_features': src_features, 'src_labels': src_labels}, os.path.join(args.output_dir, 'imagenet_val_features_labels.pkl'))
        
    if sample_per_class>0:
        src_features_sampled = []
        src_labels_sampled = []
        for i in range(src_labels.max().item()+1):
            idx = torch.where(src_labels == i)[0]
            src_features_sampled.append(src_features[idx[:10]])
            src_labels_sampled.append(src_labels[idx[:10]])
            
        src_features = torch.cat(src_features_sampled, dim=0)
        src_labels = torch.cat(src_labels_sampled, dim=0)
        
    return src_features, src_labels