# The code is modified from domainbed.algorithms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
import ot

from domainbed.algorithms import Algorithm
from domainbed import networks


ALGORITHMS = [
    'OTVP',
    'T3A', 
    'TentFull', 
    'TentPreBN',  # Tent-BN in the paper
    'TentClf',  # Tent-C in the paper
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class PrependPrompt():
    def __init__(self, network, domain_token):
        self.featurizer = network
        self.domain_tokens = domain_token
        if hasattr(self.featurizer.network, 'pos_embed'):   # ViT from timm
            self.pos_embeddings = self.featurizer.network.pos_embed
        else:   # ViT from torchvision
            self.pos_embeddings = self.featurizer.network.encoder.pos_embedding
        
    def add_domain_prompt(self):
        domain_tokens = self.domain_tokens
        def _add_domain_prompt(model, x):
            act = x[0]
            x_new = torch.cat([act, domain_tokens], dim=1)
            return (x_new, )
        return _add_domain_prompt
    
    def __enter__(self):
        if hasattr(self.featurizer.network, 'pos_embed'):
            self.hook = self.featurizer.network.norm_pre.register_forward_pre_hook(self.add_domain_prompt())
        else:
            self.hook = self.featurizer.network.encoder.dropout.register_forward_pre_hook(self.add_domain_prompt())
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
    

class OTVP(Algorithm):
    """
    OT-VP: Optimal Transport-guided Visual Prompting for Test-Time Adaptation
    """     
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(OTVP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.global_iter = 0
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        assert self.hparams['vit_base_16'] == True
        self.hidden_dim = 768
        self.prompt_dim = self.hparams['prompt_dim'] if 'prompt_dim' in hparams else 4
        self.device = hparams['device'] if 'device' in hparams else 'cuda'
        self.label_distance = self.hparams['lambda'] if 'lambda' in hparams else 1e4
        lr_prompt = self.hparams['lr_prompt'] if 'lr_prompt' in hparams else 1e-1
            
        # random initialization for prompt tokens
        self.inital = torch.empty(self.prompt_dim, self.hidden_dim).normal_(std=0.02)
        self.prompt_tokens = nn.Parameter(
            self.inital.clone().to(self.device)
        )
        
        # optimizer
        self.prompt_opt = torch.optim.AdamW(
            [self.prompt_tokens],
            lr=lr_prompt,
            weight_decay=1e-5
        )
        
    @torch.no_grad()
    def forward_feature(self, all_x_trg):
        domain_prompts = self.prompt_tokens
        with PrependPrompt(self.featurizer, domain_prompts.repeat(all_x_trg.shape[0], 1, 1)):
            all_logit_trg = self.network(all_x_trg)
        return all_logit_trg
    
    @torch.no_grad()
    def forward_pseudo_label(self, x, domain=None):
        logits = self.forward_feature(x)
        _, pseudo_label = torch.max(logits, 1)
        return pseudo_label
    
    def forward_feature_with_prompt(self, all_x_trg):
        domain_prompts = self.prompt_tokens
        with PrependPrompt(self.featurizer, domain_prompts.repeat(all_x_trg.shape[0], 1, 1)):
            all_z_trg = self.featurizer(all_x_trg)
        return all_z_trg
    
    def update(self, minibatches, precomputed_src=None):
        '''
        minibatches: SRC labeled data
        unlabeled: TGT unlabeled/labeled data
        '''
        if precomputed_src is not None:
            src_features, src_labels = precomputed_src
        self.prompt_opt.zero_grad()
        
        all_x = torch.cat([x for x,y in minibatches])

        self.network.eval()
        pseudo_labels = self.forward_pseudo_label(all_x)
        all_z_trg = self.forward_feature_with_prompt(all_x)
        
        optimal_couppling, cost_matrix = compute_conditional_wd(all_z_trg, pseudo_labels, src_features, src_labels, self.device, label_distance=self.label_distance)
        
        loss = torch.sqrt((optimal_couppling * cost_matrix).sum())
        loss.backward()
        self.prompt_opt.step()
        return {'loss': loss.item()}

    def predict(self, x, domain=None):
        return self.forward_feature(x)
   

def compute_conditional_wd(trg_features, trg_labels, src_features, src_labels, device, normalize=True, label_distance=1e4):
    """Compute conditional Wasserstein distance between source and target features."""
    src_features, src_labels = src_features.to(device), src_labels.to(device)
    if normalize:
        src_mean = src_features.mean(dim=0, keepdim=True)
        src_std = src_features.std(dim=0, keepdim=True)
        src_features = (src_features - src_mean) / src_std
        
        trg_mean = trg_features.mean(dim=0, keepdim=True)
        trg_std = trg_features.std(dim=0, keepdim=True)
        trg_features = (trg_features - trg_mean) / trg_std
        
    feature_cost = torch.cdist(src_features, trg_features, p=2) ** 2
    label_cost = (src_labels.unsqueeze(1) != trg_labels.unsqueeze(0)).float()
    cost_matrix = feature_cost + (label_cost * label_distance).to(device)
    cost_matrix_np = cost_matrix.detach().cpu().numpy()
    
    optimal_couppling_np = ot.emd(ot.unif(src_features.shape[0]), ot.unif(trg_features.shape[0]), cost_matrix_np)
    
    optimal_couppling = torch.tensor(optimal_couppling_np).to(device)
    return  optimal_couppling, cost_matrix


class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x)
        else:
            z = x
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(y_hat.device)
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


class TentFull(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=hparams['alpha'])
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        optimizer = torch.optim.Adam(
            params, 
            lr=algorithm.hparams["lr"]*alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


class TentPreBN(TentFull):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.classifier = PreBN(adapted_algorithm.classifier, adapted_algorithm.featurizer.n_outputs)
        adapted_algorithm.network = torch.nn.Sequential(adapted_algorithm.featurizer, adapted_algorithm.classifier)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.bn.parameters(), 
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer


class TentClf(TentFull):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.parameters(), 
            lr=algorithm.hparams["lr"]  * alpha,
            weight_decay=algorithm.hparams['weight_decay']
        )
        adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None   
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class PreBN(torch.nn.Module):
    def __init__(self, m, num_features, **kwargs):
        super().__init__()
        self.m = m
        self.bn = torch.nn.BatchNorm1d(num_features, **kwargs)
        self.bn.requires_grad_(True)
        self.bn.track_running_stats = False
        self.bn.running_mean = None
        self.bn.running_var = None
        
    def forward(self, x):
        x = self.bn(x)
        return self.m(x)

    def predict(self, x):
        return self(x)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


