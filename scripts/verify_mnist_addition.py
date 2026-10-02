import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mnist_addition import MNISTAdder


class PairModel(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.extractor = model.extractor
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.fc3 = model.fc3
        self.model_type = model_type

    def forward(self, images):
        left = self.extractor(images[:, 0:1])
        right = self.extractor(images[:, 1:2])
        features = torch.cat([left, right], dim=1)
        if self.model_type == 'baseline':
            output = torch.relu(self.fc1(features))
            output = torch.relu(self.fc2(output))
            return self.fc3(output)
        return features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--examples', type=int, default=20)
    parser.add_argument('--eps', type=float, nargs='+', default=[0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255])
    parser.add_argument('--output', default='verification_results.csv')
    return parser.parse_args()


def load_test_examples(count):
    dataset = torchvision.datasets.MNIST('data/mnist', train=False, download=True)
    images = dataset.data.float().div(255.0)
    labels = dataset.targets
    by_class = {digit: images[labels == digit] for digit in range(10)}

    examples = []
    for left_digit, right_digit in itertools.product(range(10), repeat=2):
        if len(examples) >= count:
            break
        examples.append((
            torch.stack([by_class[left_digit][0], by_class[right_digit][0]]),
            left_digit + right_digit,
        ))
    return torch.stack([item[0] for item in examples]), [item[1] for item in examples]


def bounds_for_inputs(prefix, images, eps):
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except ImportError as error:
        raise RuntimeError(
            'auto_LiRPA is required. Use an alpha-beta-CROWN environment '
            '(typically Python 3.10 or 3.11) and install its pinned '
            'dependencies before running verification.'
        ) from error

    bounded_prefix = BoundedModule(
        prefix,
        torch.zeros_like(images[:1]),
        device='cpu',
    )
    perturbation = PerturbationLpNorm(
        norm=np.inf,
        eps=eps,
        x_L=torch.clamp(images - eps, 0.0, 1.0),
        x_U=torch.clamp(images + eps, 0.0, 1.0),
    )
    bounded_images = BoundedTensor(images, perturbation)
    lower, upper = bounded_prefix.compute_bounds(
        x=(bounded_images,),
        method='backward',
    )
    return lower, upper


def sign_bits(values):
    return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


def symbolic_output(model, bits):
    solver_input = torch.tensor(
        [[1.0 if bit else -1.0 for bit in bits] + [0.0] * 5],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return model.sat(solver_input).squeeze(0)


def possible_bit_assignments(lower, upper):
    choices = []
    for lower_value, upper_value in zip(lower.tolist(), upper.tolist()):
        if lower_value > 0:
            choices.append([True])
        elif upper_value < 0:
            choices.append([False])
        else:
            choices.append([False, True])
    return itertools.product(*choices)


def verify_example(model, model_type, image_pair, target, eps):
    images = image_pair.unsqueeze(0)
    prefix = PairModel(model, model_type).cpu().eval()
    lower, upper = bounds_for_inputs(prefix, images, eps)

    with torch.no_grad():
        clean_features = prefix(images)
        if model_type == 'baseline':
            clean_output = clean_features
        else:
            clean_output = model((images[:, 0:1], images[:, 1:2]), return_sat=True)

    target_bits = torch.tensor(
        [float(bit) for bit in format(target, '05b')],
        dtype=torch.float32,
    )
    clean_correct = bool(torch.equal((clean_output.squeeze(0) >= 0).float(), target_bits))

    if model_type == 'baseline':
        clean_sign = sign_bits(clean_output.squeeze(0))
        stable = bool(torch.all(
            ((clean_sign > 0) & (lower.squeeze(0) > 0))
            | ((clean_sign < 0) & (upper.squeeze(0) < 0))
        ))
        certified_correct = stable and clean_correct
        ambiguous_bits = int(((lower <= 0) & (upper >= 0)).sum().item())
    else:
        clean_bits = sign_bits(clean_features.squeeze(0)) > 0
        assignments = list(possible_bit_assignments(lower.squeeze(0), upper.squeeze(0)))
        clean_symbolic = (clean_output.squeeze(0) >= 0)
        all_same = True
        all_correct = True
        for assignment in assignments:
            output = symbolic_output(model, assignment)
            output_bits = output >= 0
            all_same = all_same and bool(torch.equal(output_bits, clean_symbolic))
            all_correct = all_correct and bool(torch.equal(output_bits, target_bits))
        stable = all_same
        certified_correct = all_correct
        ambiguous_bits = int(((lower <= 0) & (upper >= 0)).sum().item())

    return {
        'eps': eps,
        'clean_correct': clean_correct,
        'certified_stable': stable,
        'certified_correct': certified_correct,
        'ambiguous_grounding_bits': ambiguous_bits,
    }


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_type = checkpoint['model_type']
    if model_type not in ('smt', 'baseline'):
        raise ValueError('Unsupported checkpoint model_type: {}'.format(model_type))
    if model_type == 'smt' and checkpoint.get('maxsat_forward', False):
        raise ValueError('This verifier currently supports ordinary SMT forward passes, not MaxSMT.')

    model = MNISTAdder(
        use_maxsmt=checkpoint.get('maxsat_backward', False),
    ).cpu().eval()
    model.load_state_dict(checkpoint['model_state'])
    images, targets = load_test_examples(args.examples)

    rows = []
    for eps in args.eps:
        results = [
            verify_example(model, model_type, image, target, eps)
            for image, target in zip(images, targets)
        ]
        row = {
            'checkpoint': args.checkpoint,
            'model_type': model_type,
            'eps': eps,
            'examples': len(results),
            'clean_accuracy': sum(result['clean_correct'] for result in results) / len(results),
            'certified_stability': sum(result['certified_stable'] for result in results) / len(results),
            'certified_accuracy': sum(result['certified_correct'] for result in results) / len(results),
            'mean_ambiguous_bits': sum(result['ambiguous_grounding_bits'] for result in results) / len(results),
        }
        rows.append(row)
        print(row)

    with open(args.output, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('saved {}'.format(args.output))


if __name__ == '__main__':
    main()
