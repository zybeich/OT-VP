

import argparse
import collections
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import sys
import time
import uuid
import ot

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision

from domainbed import algorithms, datasets, hparams_registry, adapt_algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.torchmisc import dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test-Time Adaptation')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="tta",
        choices=["domain_generalization", "domain_adaptation", "tta"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=101,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--train_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[-1])
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--corruption', type=int, default=None)
    parser.add_argument('--severity', type=int, default=5)
    args = parser.parse_args()
    
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
        
        
    hparams['data_augmentation'] = False
    

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs + args.train_envs, hparams)


    num_workers_eval =  8
    batch_size_eval = 128
    hparams['device'] = device
    eval_class =  dataloader
    
    corruptions = ['gaussian_noise',
                    'shot_noise',
                    'impulse_noise',
                    'defocus_blur',
                    'glass_blur',
                    'motion_blur',
                    'zoom_blur',
                    'snow',
                    'frost',
                    'fog',
                    'brightness',
                    'contrast',
                    'elastic_transform',
                    'pixelate',
                    'jpeg_compression']
    corruption = corruptions[args.corruption]
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    print(f'Loading Corrupted data from {f"ImageNet-C/{corruptions[args.corruption]}/{args.severity}"}')
    trg_dataset = ImageFolder(os.path.join(args.data_dir, f'ImageNet-C/{corruptions[args.corruption]}/{args.severity}'),  transform=transform)
    src_dataset = ImageFolder(os.path.join(args.data_dir, "ImageNet/val"), transform=transform)
    
    src_test_loaders = [FastDataLoader(
            dataset=src_dataset,
            batch_size=num_workers_eval,
            num_workers=num_workers_eval)]
    
    
    algorithm_class = adapt_algorithms.get_algorithm_class(args.algorithm)
    tta = algorithm_class( (3, 224, 224,), 1000,
        1, hparams)
    tta.to(device)
        
    if args.dataset.lower().startswith("imagenet"): # load from timm
        import timm
        ckpt = timm.create_model("vit_base_patch16_224", pretrained=True).head.state_dict()
        missing_keys, unexpected_keys = tta.classifier.load_state_dict(ckpt, strict=False)
        print("ImageNet pre-trained model restored from timm successfully")
        # print("missing keys: {}".format(missing_keys))
        # print("unexpected keys: {}".format(unexpected_keys))

    # Load offline computed source features
    src_features, src_labels = misc.load_src_features(args, tta, src_test_loaders, device)
    print(f"Pre-computed source features loaded successfully")
    
    # target dataset
    trg_test_loaders = [InfiniteDataLoader(
        dataset=trg_dataset,
        weights=None,
        batch_size=batch_size_eval,
        num_workers=dataset.N_WORKERS)
        ]
    trg_test_iterator = zip(*trg_test_loaders)
    
    trg_eval_loader = FastDataLoader(
            dataset=trg_dataset,
            batch_size=batch_size_eval,
            num_workers=dataset.N_WORKERS)
    
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = len(trg_dataset)/batch_size_eval 
    n_steps = args.steps
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    last_results_keys = None
    start_time = time.time()
    print('='*20+'START'+'='*20)
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(trg_test_iterator)]
        
        step_vals = tta.update(minibatches_device, (src_features, src_labels))

        # evaluate on the test set
        if ((step % checkpoint_freq == 0) or (step == n_steps - 1)) and (step > 0):
            checkpoint_vals['adapt_time'].append(time.time() - start_time)
            results = {
                'step': step,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            acc = misc.accuracy(tta, trg_eval_loader, None, device, name=None, domain=None)
            results[f'{corruption[:4]}_{args.severity}_acc'] = acc
            
            
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            checkpoint_vals = collections.defaultdict(lambda: [])

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    print('='*21+'END'+'='*21)
    print(f'Total Time: {time.time() - start_time:.1f}s')
    
    torch.cuda.empty_cache()
