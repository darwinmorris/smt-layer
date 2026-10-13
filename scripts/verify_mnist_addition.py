import argparse
import csv
import itertools
import os
import sys

import sympy
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mnist_addition import MNISTAdder
from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints, input_vars, output_vars


class PairModel(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        extractor = model.extractor
        self.conv1 = nn.Conv2d(2, 128, 3, 2)
        self.conv2 = nn.Conv2d(128, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.extractor_fc1 = nn.Linear(12544, 512)
        self.extractor_fc2 = nn.Linear(512, 8)
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.fc3 = model.fc3
        self.model_type = model_type

        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.weight[:64, 0].copy_(extractor.conv1.weight[:, 0])
            self.conv1.weight[64:, 1].copy_(extractor.conv1.weight[:, 0])
            self.conv1.bias.copy_(extractor.conv1.bias.repeat(2))
            self.conv2.weight.zero_()
            self.conv2.weight[:64, :64].copy_(extractor.conv2.weight)
            self.conv2.weight[64:, 64:].copy_(extractor.conv2.weight)
            self.conv2.bias.copy_(extractor.conv2.bias.repeat(2))
            self.conv3.weight.zero_()
            self.conv3.weight[:128, :64].copy_(extractor.conv3.weight)
            self.conv3.weight[128:, 64:].copy_(extractor.conv3.weight)
            self.conv3.bias.copy_(extractor.conv3.bias.repeat(2))
            self.conv4.weight.zero_()
            self.conv4.weight[:128, :128].copy_(extractor.conv4.weight)
            self.conv4.weight[128:, 128:].copy_(extractor.conv4.weight)
            self.conv4.bias.copy_(extractor.conv4.bias.repeat(2))
            self.extractor_fc1.weight.zero_()
            self.extractor_fc1.bias.copy_(extractor.fc1.bias.repeat(2))
            self.extractor_fc1.weight[:256, :6272].copy_(extractor.fc1.weight)
            self.extractor_fc1.weight[256:, 6272:].copy_(extractor.fc1.weight)
            self.extractor_fc2.weight.zero_()
            self.extractor_fc2.bias.copy_(extractor.fc2.bias.repeat(2))
            self.extractor_fc2.weight[:4, :256].copy_(extractor.fc2.weight)
            self.extractor_fc2.weight[4:, 256:].copy_(extractor.fc2.weight)

    def forward(self, images):
        # Block-diagonal weights instead of images[:, 0:1] slicing: auto_LiRPA's
        # BoundSlice asserts on sparse (unstable_idx) A matrices in patches mode.
        output = torch.relu(self.conv1(images))
        output = torch.relu(self.conv2(output))
        output = torch.relu(self.conv3(output))
        output = torch.relu(self.conv4(output))
        output = output.flatten(1)
        output = torch.relu(self.extractor_fc1(output))
        features = self.extractor_fc2(output)
        if self.model_type == 'baseline':
            output = torch.relu(self.fc1(features))
            output = torch.relu(self.fc2(output))
            return self.fc3(output)
        return features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--examples', type=int, default=20)
    parser.add_argument('--eps', type=float, nargs='+', default=[0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 0.05])
    parser.add_argument('--output', default='verification_results.csv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=60, help='BaB timeout per verify call, in seconds')
    parser.add_argument('--pgd-steps', type=int, default=100)
    parser.add_argument('--pgd-restarts', type=int, default=30)
    # The model's clean prediction (z3) always runs on CPU; this is only for abcrown.
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_test_examples(count, seed):
    dataset = torchvision.datasets.MNIST('data/mnist', train=False, download=True)
    images = dataset.data.float().div(255.0)
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(len(images), (count,), generator=generator)
    right = torch.randint(len(images), (count,), generator=generator)
    pairs = torch.stack([images[left], images[right]], dim=1)
    targets = (dataset.targets[left] + dataset.targets[right]).tolist()
    return pairs, targets


def decode_sum(model_type, bits):
    if model_type == 'baseline':
        return int(''.join(map(str, bits)), 2)
    # The SMT layer reads features 0-3 and 4-7 as 4-bit numbers (MSB first) and adds them.
    return int(''.join(map(str, bits[:4])), 2) + int(''.join(map(str, bits[4:])), 2)


def wrong_sum_clauses(model_type, total):
    # abcrown negates a DSL output constraint and expands it to DNF, which blows up
    # for "sum == total" over 8 features. Instead, build the violation DNF directly:
    # the minimized set of sign patterns whose decoded sum is not `total`. Each
    # clause is (C, rhs) with C @ y < rhs; row +e_j means y_j < 0, -e_j means y_j > 0.
    num_bits = 5 if model_type == 'baseline' else 8
    symbols = sympy.symbols('b0:{}'.format(num_bits))
    wrong = [
        list(bits) for bits in itertools.product([0, 1], repeat=num_bits)
        if decode_sum(model_type, bits) != total
    ]
    cover = sympy.SOPform(symbols, wrong)
    clauses = []
    for term in sympy.Or.make_args(cover):
        rows = []
        for literal in sympy.And.make_args(term):
            row = torch.zeros(num_bits)
            if isinstance(literal, sympy.Not):
                row[symbols.index(literal.args[0])] = 1.0
            else:
                row[symbols.index(literal)] = -1.0
            rows.append(row)
        clauses.append([(torch.stack(rows), torch.zeros(len(rows)))])
    return clauses


def bits_to_int(values):
    return int(''.join('1' if value >= 0 else '0' for value in values.tolist()), 2)


def verify_example(solver, model_type, image_pair, clean_sum, target, eps):
    clauses = wrong_sum_clauses(model_type, clean_sum)
    lower = torch.clamp(image_pair - eps, 0.0, 1.0).repeat(len(clauses), 1, 1, 1)
    upper = torch.clamp(image_pair + eps, 0.0, 1.0).repeat(len(clauses), 1, 1, 1)
    constraints = IOConstraints(lower=lower, upper=upper, clauses=clauses)
    # Proving the clean sum is stable also proves correctness when the clean
    # prediction is right; when it is wrong, the clean point is a counterexample.
    result = solver.verify(constraints=constraints)
    # result.success is also True for 'falsified' (it only means "not unknown").
    stable = result.status == 'verified'
    clean_correct = clean_sum == target
    return {
        'clean_correct': clean_correct,
        'certified_stable': stable,
        'certified_correct': stable and clean_correct,
        'status': result.status,
    }


def main():
    args = parse_args()
    print('Verifying on {}'.format(args.device))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_type = checkpoint['model_type']
    if model_type not in ('smt', 'baseline'):
        raise ValueError('Unsupported checkpoint model_type: {}'.format(model_type))
    if model_type == 'smt' and checkpoint.get('maxsat_forward', False):
        raise ValueError('This verifier currently supports ordinary SMT forward passes, not MaxSMT.')

    model = MNISTAdder(use_maxsmt=checkpoint.get('maxsat_backward', False)).eval()
    model.load_state_dict(checkpoint['model_state'])
    images, targets = load_test_examples(args.examples, args.seed)
    with torch.no_grad():
        clean_outputs = model((images[:, 0:1], images[:, 1:2]), return_sat=model_type == 'smt')
    clean_sums = [bits_to_int(output) for output in clean_outputs]

    x = input_vars((2, 28, 28))
    y = output_vars(5 if model_type == 'baseline' else 8)
    config = (
        ConfigBuilder.from_defaults()
        .set('general/device', args.device)
        .set('bab/timeout', args.timeout)
        .set('attack/pgd_steps', args.pgd_steps)
        .set('attack/pgd_restarts', args.pgd_restarts)
    )
    solver = ABCrownSolver(PairModel(model, model_type).eval(), x, y, config=config)

    # Eps boxes are nested, so a counterexample found at one eps lies inside every
    # larger box too; once an example is unsafe, later eps reuse that result.
    unsafe_results = {}
    rows = []
    for eps in sorted(args.eps):
        results = []
        for index, (image, clean_sum, target) in enumerate(zip(images, clean_sums, targets)):
            if index in unsafe_results:
                results.append(unsafe_results[index])
                continue
            result = verify_example(solver, model_type, image, clean_sum, target, eps)
            if result['status'] == 'falsified':
                unsafe_results[index] = result
            results.append(result)
        row = {
            'checkpoint': args.checkpoint,
            'model_type': model_type,
            'eps': eps,
            'examples': len(results),
            'clean_accuracy': sum(result['clean_correct'] for result in results) / len(results),
            'certified_stability': sum(result['certified_stable'] for result in results) / len(results),
            'certified_accuracy': sum(result['certified_correct'] for result in results) / len(results),
            # Upper bound on robust accuracy: correct examples the attack did not break.
            'empirical_accuracy': sum(
                result['clean_correct'] and result['status'] != 'falsified' for result in results
            ) / len(results),
            'statuses': ';'.join(sorted(set(result['status'] for result in results))),
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
