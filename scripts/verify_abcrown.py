"""alpha-beta-CROWN robustness verification for MNIST addition, with resumable CSV results.

One abcrown verify() call per (example, eps): PGD attack, alpha-CROWN, then BaB on
the exact property "the decoded sum does not change". For SMT/digit models the
property is stated over the 8 extractor logits, whose signs fully determine the
SMT layer's output.

Rerun the same command after a crash or Ctrl-C to resume; completed tasks are kept
in <output>.resume.json and per-example rows in <output stem>_examples.csv.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--examples', type=int, default=100)
    parser.add_argument('--eps', type=float, nargs='+', default=[0.0, 1 / 255, 2 / 255, 3/255,  4 / 255, 5/255, 6/255, 8 / 255])
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
    return pairs, targets, left.tolist(), right.tolist()


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
    # All clauses form one OR list for a single input box.
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
        clauses.append((torch.stack(rows), torch.zeros(len(rows))))
    return clauses


def bits_to_int(values):
    return int(''.join('1' if value >= 0 else '0' for value in values.tolist()), 2)


def verify_example(solver, model_type, image_pair, clean_sum, target, eps):
    clauses = wrong_sum_clauses(model_type, clean_sum)
    lower = torch.clamp(image_pair - eps, 0.0, 1.0).unsqueeze(0)
    upper = torch.clamp(image_pair + eps, 0.0, 1.0).unsqueeze(0)
    constraints = IOConstraints(lower=lower, upper=upper, clauses=[clauses])
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
        'clauses': len(clauses),
    }


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_write(path, write):
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'w', newline='') as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def save_csv(path, rows):
    if not rows:
        return
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, write)


def summarize(checkpoints, eps_values, records):
    rows = []
    for number, checkpoint in enumerate(checkpoints):
        for eps in eps_values:
            results = [r for r in records.values()
                       if r['checkpoint_index'] == number and r['eps'] == eps]
            if not results:
                continue
            n = len(results)
            rows.append({
                'checkpoint': checkpoint['path'],
                'model_type': checkpoint['model_type'],
                'eps': eps,
                'examples': n,
                'clean_accuracy': sum(r['clean_correct'] for r in results) / n,
                'certified_stability': sum(r['certified_stable'] for r in results) / n,
                'certified_accuracy': sum(r['certified_correct'] for r in results) / n,
                # Upper bound on robust accuracy: correct examples the attack did not break.
                'empirical_accuracy': sum(
                    r['clean_correct'] and r['status'] != 'falsified' for r in results) / n,
                'verified': sum(r['status'] == 'verified' for r in results),
                'falsified': sum(r['status'] == 'falsified' for r in results),
                'unknown': sum(r['status'] == 'unknown' for r in results),
                'errors': sum(r['status'] == 'error' for r in results),
                'total_seconds': sum(r['seconds'] for r in results),
            })
    return rows


def load_checkpoint(path, images):
    """Load a checkpoint, check its PairModel rewrite, and return clean sums."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model_type = checkpoint['model_type']
    if model_type not in ('smt', 'digit', 'baseline'):
        raise ValueError('Unsupported checkpoint model_type: {}'.format(model_type))
    if model_type != 'baseline' and checkpoint.get('maxsat_forward', False):
        raise ValueError('This verifier currently supports ordinary SMT forward passes, not MaxSMT.')
    use_sat = model_type != 'baseline'

    model = MNISTAdder(use_maxsmt=checkpoint.get('maxsat_backward', False)).eval()
    model.load_state_dict(checkpoint['model_state'])
    pair_model = PairModel(model, model_type).eval()
    with torch.no_grad():
        split = (images[:, 0:1], images[:, 1:2])
        clean_outputs = model(split, return_sat=use_sat)
        # The block-diagonal rewrite must reproduce the original network exactly.
        reference = model(split, return_feats=True) if use_sat else clean_outputs
        if not torch.allclose(pair_model(images), reference, atol=1e-4):
            raise AssertionError('PairModel does not match the original model: {}'.format(path))
    return model_type, pair_model, [bits_to_int(output) for output in clean_outputs]


def main():
    args = parse_args()
    print('Verifying on {}'.format(args.device))
    images, targets, left, right = load_test_examples(args.examples, args.seed)
    checkpoints = []
    for path in args.checkpoint:
        path = str(Path(path).resolve())
        model_type, pair_model, clean_sums = load_checkpoint(path, images)
        checkpoints.append({'path': path, 'sha256': digest(path), 'model_type': model_type,
                            'pair_model': pair_model, 'clean_sums': clean_sums})

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix('.resume.json')
    details_path = output.with_name(output.stem + '_examples.csv')
    eps_values = sorted(args.eps)
    manifest = {
        'version': 2,
        'script_sha256': digest(__file__),
        'checkpoints': [{k: c[k] for k in ['path', 'sha256', 'model_type']} for c in checkpoints],
        'settings': {k: getattr(args, k) for k in
                     ['examples', 'seed', 'timeout', 'pgd_steps', 'pgd_restarts', 'device']},
        'eps': eps_values,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['manifest'] != manifest:
            raise ValueError('Resume settings, code, or checkpoints differ. Use a new --output filename.')
        print('Resuming: {} tasks already saved.'.format(len(state['records'])), flush=True)
    else:
        if output.exists() or details_path.exists():
            raise ValueError('Output already exists without matching resume state. Use a new --output filename.')
        state = {'manifest': manifest, 'records': {}}
    records = state['records']

    def save():
        atomic_write(state_path, lambda stream: json.dump(state, stream, indent=2))
        save_csv(details_path, list(records.values()))
        save_csv(output, summarize(checkpoints, eps_values, records))

    x = input_vars((2, 28, 28))
    config = (
        ConfigBuilder.from_defaults()
        .set('general/device', args.device)
        .set('general/seed', args.seed)
        .set('bab/timeout', args.timeout)
        .set('attack/pgd_steps', args.pgd_steps)
        .set('attack/pgd_restarts', args.pgd_restarts)
        # Bound first (CROWN/alpha-CROWN); attack only if that is inconclusive, then BaB.
        .set('attack/pgd_order', 'after')
    )

    try:
        for number, checkpoint in enumerate(checkpoints):
            model_type = checkpoint['model_type']
            y = output_vars(5 if model_type == 'baseline' else 8)
            solver = ABCrownSolver(checkpoint['pair_model'], x, y, config=config)
            for eps in eps_values:
                for index, (image, clean_sum, target) in enumerate(
                        zip(images, checkpoint['clean_sums'], targets)):
                    key = '{}:{}:{!r}'.format(number, index, eps)
                    if key in records:
                        continue
                    # Eps boxes are nested, so a counterexample found at a smaller eps lies
                    # inside this box too; reuse that result instead of re-verifying.
                    earlier = next((r for r in records.values()
                                    if r['checkpoint_index'] == number and r['example_index'] == index
                                    and r['eps'] < eps and r['status'] == 'falsified'), None)
                    print('{} eps={:.8g} example {}/{}'.format(
                        model_type, eps, index + 1, len(images)), flush=True)
                    start = time.monotonic()
                    if earlier is not None:
                        result = {k: earlier[k] for k in ['clean_correct', 'certified_stable',
                                                          'certified_correct', 'status', 'clauses']}
                        result['falsified_at_eps'] = earlier['falsified_at_eps']
                    else:
                        try:
                            result = verify_example(solver, model_type, image, clean_sum, target, eps)
                        except Exception as exc:
                            traceback.print_exc()
                            result = {'clean_correct': clean_sum == target, 'certified_stable': False,
                                      'certified_correct': False, 'status': 'error', 'clauses': '',
                                      'error': str(exc)}
                        result['falsified_at_eps'] = eps if result['status'] == 'falsified' else ''
                    records[key] = {
                        'checkpoint_index': number, 'checkpoint': checkpoint['path'],
                        'model_type': model_type, 'example_index': index,
                        'left_mnist_index': left[index], 'right_mnist_index': right[index],
                        'target': target, 'clean_sum': clean_sum, 'eps': eps, 'error': '',
                        **result, 'seconds': time.monotonic() - start,
                    }
                    save()
                    print('  {} in {:.1f}s; saved'.format(result['status'], records[key]['seconds']),
                          flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted; rerun the same command to resume completed tasks.', flush=True)
    save()
    for row in summarize(checkpoints, eps_values, records):
        print(row)
    print('Results: {}; examples: {}'.format(output, details_path))


if __name__ == '__main__':
    main()
