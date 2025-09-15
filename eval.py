from pathlib import Path
import argparse
import random
from typing import Union, Callable, Tuple
from collections import Counter

import torch
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             top_k_accuracy_score, roc_auc_score, roc_curve)
from sklearn.utils import check_random_state

from src.face_embedder.vision_transformer import vit_base

# --- Utilities/shared constants
METRICS = ("mean_accuracy", "top5_accuracy", "balanced_accuracy", "precision", "recall", "f1")


class PetFaceChimpStar(torch.utils.data.Dataset):
    '''Subset of chimps from the PetFace dataset with additional filtering for duplicates and low-quality items.'''

    ORIG_IMG_SIZE: int = 224
    CLASS: str = 'chimp'

    def __init__(
        self,
        img_size: Union[int, Tuple] = ORIG_IMG_SIZE,
        root: str = None,
        pad_around_bbox: float = None,
        transforms: Union[Callable, None] = None,
        target_transforms: Union[Callable, None] = None,
        return_dict: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.return_dict = return_dict
        self.pad_around_bbox = pad_around_bbox
        self.img_size = [img_size, img_size] if isinstance(img_size, int) else img_size
        self.seed = seed

        # Handle transforms
        self.transforms = transforms
        self.target_transforms = target_transforms
        # doing resize here because it is not possible to do it in the transforms
        # hacking resizing to the transforms: i don't want to change the logic of transforms for now, and this
        # will do it similar to the LomaMt class
        if img_size != self.ORIG_IMG_SIZE:
            if self.transforms is None:
                self.transforms = T.Resize(self.img_size)
            else:
                self.transforms = T.Compose([T.Resize(self.img_size), self.transforms])

        # Deduce dataset paths
        dataset = list((self.root / self.CLASS).glob('**/*.png')) + list((self.root / self.CLASS).glob('**/*.jpg'))
        dataset = sorted(dataset)
        # load the filtered rel_paths
        self.path_to_filtered = Path(root) / 'filtered_petface_files.txt'
        filtered = set([line.strip() for line in open(self.path_to_filtered, 'r').readlines()])
        # filter to make the `chimp_star` dataset
        dataset = [p for p in dataset if '/'.join(p.parts[-3:]) not in filtered]

        # add labels from the path to the dataset
        dataset = [(p, p.parts[-2]) for p in dataset]

        self.num_classes = len(set([p[1] for p in dataset]))
        self.dataset = dataset

    def __getitem__(self, index: int):
        img_path, padded_label = self.dataset[index]
        img = self.get_image(img_path)

        # Apply transforms if available
        if self.transforms is not None:
            img = self.transforms(img)

        label = int(padded_label)
        if self.return_dict:
            return {'image': img, 'meta': {'label': label, 'path': str(img_path)}, }
        else:
            return img, label

    def get_image(self, img_path: Path) -> Image:
        '''Loads an image from an image path (jpeg, png, etc.) using pil.'''
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.dataset)


def softmax_weight_factory(T: float = 0.07):
    def _weights(distances):
        # distances: (n_queries, k) — cosine *distance* expected in [0, 2]
        sims = 1.0 - distances
        z = sims / T
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    return _weights


def safe_topk_from_knn(clf: KNeighborsClassifier, X_test, y_true, k=5):
    """Gracefully compute top-k accuracy (handles <k classes, no-proba cases)."""
    if not hasattr(clf, "predict_proba"):
        return np.nan
    y_prob = clf.predict_proba(X_test)
    if y_prob is None or y_prob.size == 0:
        return np.nan
    actual_k = min(k, y_prob.shape[1])
    if actual_k < 1:
        return np.nan
    # Align labels with proba columns
    return top_k_accuracy_score(y_true, y_prob, k=actual_k, labels=clf.classes_)

def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def knn_sweep(X_train, y_train, X_test, y_test, cfg, reduce_k_if_imbalance=True):
    # 'uniform' | 'distance' | 'softmax' | callable.
    weights = softmax_weight_factory() if cfg['weight_scheme'] == 'softmax' else cfg['weight_scheme']
    min_per_class = min(Counter(y_train).values())

    results = {}
    for k in cfg['ks']:
        k_eff = min(k, min_per_class) if reduce_k_if_imbalance else k
        # Skip redundant evaluations if k was already reduced earlier
        if k_eff in results and k_eff != k:
            continue

        clf = KNeighborsClassifier(n_neighbors=k_eff, metric=cfg['distance_metric'], weights=weights)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test) # Top-1 predictions

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mean_acc = accuracy_score(y_test, y_pred) # This is Top-1 accuracy
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        top5_acc = safe_topk_from_knn(clf, X_test, y_test, k=5)

        results[k_eff] = {
            "mean_accuracy": mean_acc,
            "top5_accuracy": top5_acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return results


def leave_one_per_class_split(X, y, *, random_state=None, return_indices=False):
    y_arr = np.asarray(y)
    rng = check_random_state(random_state)

    classes, y_enc = np.unique(y_arr, return_inverse=True)
    idxs_per_class = [np.where(y_enc == k)[0] for k in range(len(classes))]

    rare = [cls for cls, idxs in zip(classes, idxs_per_class) if len(idxs) < 2]
    if rare:
        raise ValueError(f"Cannot leave one out for classes with <2 samples: {rare}")

    test_mask = np.zeros_like(y_arr, dtype=bool)
    for idxs in idxs_per_class:
        test_mask[rng.choice(idxs)] = True

    if return_indices:
        return (~test_mask).nonzero()[0], test_mask.nonzero()[0]

    X = np.asarray(X)
    return X[~test_mask], X[test_mask], y_arr[~test_mask], y_arr[test_mask]



def compute_verification_distances(X, y, distance_metric='cosine', n_pairs_per_class=None, random_state=None):
    from sklearn.metrics.pairwise import pairwise_distances

    rng = check_random_state(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    # Get unique classes and their indices
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    positive_pairs = []
    negative_pairs = []

    # Generate positive pairs (same class)
    for cls in classes:
        indices = class_indices[cls]
        if len(indices) < 2:
            continue

        # Generate all possible pairs within class or sample n_pairs_per_class
        if n_pairs_per_class is None:
            # All possible pairs
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    positive_pairs.append((indices[i], indices[j]))
        else:
            # Sample n_pairs_per_class random pairs
            n_possible_pairs = len(indices) * (len(indices) - 1) // 2
            n_to_sample = min(n_pairs_per_class, n_possible_pairs)

            all_pairs = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
            if len(all_pairs) > 0:
                sampled_pairs = rng.choice(len(all_pairs), size=n_to_sample, replace=False)
                positive_pairs.extend([all_pairs[idx] for idx in sampled_pairs])

    # Generate negative pairs (different classes)
    n_positive = len(positive_pairs)
    n_negative_target = n_positive  # Balance positive and negative pairs

    # Sample negative pairs
    for _ in range(n_negative_target):
        # Sample two different classes
        if len(classes) < 2:
            break
        cls1, cls2 = rng.choice(classes, size=2, replace=False)
        assert cls1 != cls2, "Classes must be different for negative pairs."

        # Sample one index from each class
        idx1 = rng.choice(class_indices[cls1])
        idx2 = rng.choice(class_indices[cls2])

        negative_pairs.append((idx1, idx2))

    # Combine all pairs
    all_pairs = positive_pairs + negative_pairs
    y_pairs = np.concatenate([np.ones(len(positive_pairs)), np.zeros(len(negative_pairs))])

    # Shuffle pairs
    shuffle_indices = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in shuffle_indices]
    y_pairs = y_pairs[shuffle_indices]

    # Compute pairwise distances
    distances = []
    for i, j in all_pairs:
        if distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            # Cosine similarity = dot product for normalized vectors
            dist = 1 - np.dot(X[i], X[j])  # will 1-dist when computing similarities later
        elif distance_metric == 'euclidean':
            dist = np.linalg.norm(X[i] - X[j])
        else:
            # Use sklearn for other metrics; NOTE: features are normalized here
            dist = pairwise_distances(X[i:i+1], X[j:j+1], metric=distance_metric)[0, 0]
        distances.append(dist)

    distances = np.array(distances)

    return distances, y_pairs


def run_verification(X, y, cfg, run):
    # Compute distances and labels for verification
    distances, y_pairs = compute_verification_distances(X, y, cfg['distance_metric'])

    # For verification, we want to predict if two samples are from the same class
    # Lower distance should indicate same class (positive), so we need to negate distances
    # or use (1 - distances) for similarity scores
    if cfg['distance_metric'] == 'cosine':
        similarities = 1 - distances  # Convert cosine distance to cosine similarity
    else:
        similarities = -distances  # For other metrics, just negate

    # Compute ROC-AUC
    try:
        # ----- ROC-AUC
        roc_auc = roc_auc_score(y_pairs, similarities)
        fpr, tpr, thresholds = roc_curve(y_pairs, similarities, pos_label=1)
        # Compute EER (Equal Error Rate)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))   # point where FPR ≈ FNR
        eer = fpr[eer_idx]
        eer_thr = thresholds[eer_idx]
        # Compute TPR at specific FPR lines
        tpr_at_fpr_lines = []
        for target_fpr in (0.001, 0.01, 0.05, 0.10):      # 0.1 %, 1 %, 5 %, 10 %
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr_lines.append(
                f"  TPR@FPR={target_fpr:.3f}: {tpr[idx]:.2%} (threshold={thresholds[idx]:.5f})"
            )
        # Combine all metrics into one log message
        print(f"[Run {run}]")
        print(f"  Verification ROC-AUC: {roc_auc:.4f}\n"
            f"  EER: {eer:.3%} @ threshold {eer_thr:.5f}\n"
            + "\n".join(tpr_at_fpr_lines)
        )
    except ValueError as e:
        print(f"Could not compute ROC-AUC: {e}")
        roc_auc = np.nan

    return roc_auc


def run_verification_repeats(X, y, cfg):
    verification_results = []
    for i in range(cfg['n_repeats']):
        roc_auc = run_verification(X, y, cfg, i)
        verification_results.append(roc_auc)

    # Aggregate and log verification results
    valid_aucs = [auc for auc in verification_results if not np.isnan(auc)]
    if valid_aucs:
        mean_auc = np.mean(valid_aucs)
        if len(valid_aucs) > 1:
            ci_auc = stats.sem(valid_aucs) * stats.t.ppf(0.975, len(valid_aucs)-1)
        else:
            ci_auc = 0
        print(f"[Summary of {len(valid_aucs)} runs] Verification ROC-AUC: {mean_auc:.4f} ± {ci_auc:.4f}")
    else:
        print("No valid ROC-AUC scores computed.")

    return verification_results


def get_model(cfg):
    if cfg['model_type'] == 'dinov2':
        fix_random_seeds(cfg['seed'])
        # Load model
        model = vit_base(
            img_size = cfg['input_size'][0],
            patch_size = 14,
            init_values = 1e-05,
            ffn_layer = 'mlp',
            block_chunks = 4,
            qkv_bias = True,
            proj_bias = True,
            ffn_bias = True,
            num_register_tokens = 0,
            interpolate_offset = 0.1,
            interpolate_antialias = False,
        )
        state_dict = torch.load(cfg['pretrained_weights'], map_location="cpu", weights_only=False)['teacher']
        # Remove 'backbone.' prefix if it exists
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(cfg['device'])
        model.eval()
    elif cfg['model_type'] in ('miewid-msv2', 'miewid-msv3'):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(cfg['model_tag'], trust_remote_code=True)
    elif cfg['model_type'] == 'MegaDescriptor-L-384':
        import timm
        model = timm.create_model(cfg['model_tag'], pretrained=True)
    else:
        raise ValueError(f'Unknown model type: {cfg["model_type"]}')
    model = model.to(cfg['device'])
    model.eval()

    # Define image transforms
    img_transforms = T.Compose([
        T.Resize(cfg['input_size'][0], interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(cfg['input_size'][0]),
        T.ToTensor(),
        T.Normalize(mean=cfg['mean'], std=cfg['std']),
    ])
    return model, img_transforms

def get_features_from_dataset(dataset, model, cfg):
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['batch_size'], shuffle=False,
                                         num_workers=cfg['num_workers'])
    X = []
    y = []
    with torch.no_grad():
        for i, (image, target) in enumerate(tqdm(loader)):
            image = image.to(cfg['device'])
            emb = model(image)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            X.append(emb.cpu().numpy())
            y.append(target.cpu().numpy())
    return np.concatenate(X), np.concatenate(y)


def aggregate_and_print_knn_results(results, cfg):
    """results: list[dict[k_eff -> metrics dict]]"""
    for k in cfg['ks']:
        rows = [r[k] for r in results if k in r]
        if not rows:
            continue
        print(f"[KNN, k={k}]")
        for m in METRICS:
            vals = [row[m] for row in rows if not (isinstance(row[m], float) and np.isnan(row[m]))]
            if len(vals) == 0:
                print(f" {m}: n/a")
            else:
                vals = np.array(vals)
                # Two-sided 95% CI half-width using Student t
                ci95 = stats.sem(vals) * stats.t.ppf(0.975, vals.size - 1)
                print(f" {m}: {np.mean(vals):.4f} ± {ci95:.4f}")

def run_nrepeats_on_dataset(cfg):
    model, img_transforms = get_model(cfg)
    # log number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {cfg['model_type']} has {num_params:,} trainable parameters.")

    ds = PetFaceChimpStar(
        root=cfg['data_root'],
        img_size=[cfg['input_size'][0], cfg['input_size'][0]],
        transforms=img_transforms,
        return_dict=False
    )
    X, y = get_features_from_dataset(ds, model, cfg)

    # remove all elements that have less than `min_samples`
    unique_y, label_counts = np.unique(y, return_counts=True)
    rare_y = unique_y[label_counts < cfg['min_samples']]
    rare_mask = np.isin(y, rare_y)
    X_no_rares = X[~rare_mask]
    y_no_rares = y[~rare_mask]
    label2target = {label: i for i, label in enumerate(np.unique(y_no_rares))}
    print(f'Keeping {len(label2target)} classes, removing {len(rare_y)} rare classes.')
    y_no_rares = np.array([label2target[label] for label in y_no_rares])
    X, y = X_no_rares, y_no_rares

    # run re-id: knn sweep with leave-one-per-class cross-validation
    results = []
    print('==== Re-ID ====')
    for seed in range(cfg['seed'], cfg['seed']+cfg['n_repeats']):
        # repeat_metrics = {k: {"mean_accuracy": [], "top5_accuracy": [], "balanced_accuracy": [], "precision": [], "recall": [], "f1": []} for k in cfg['ks']}
        X_train, X_test, y_train, y_test = leave_one_per_class_split(X, y, random_state=seed)
        results.append(knn_sweep(X_train, y_train, X_test, y_test, cfg, reduce_k_if_imbalance=False))
    aggregate_and_print_knn_results(results, cfg)

    print('==== Verification ====')
    verification_results = run_verification_repeats(X, y, cfg)
    return results, verification_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='dinov2',
                        choices=['miewid-msv2', 'miewid-msv3', 'MegaDescriptor-L-384', 'dinov2'],
                        help='Type of model to use.')
    parser.add_argument('--pretrained_weights', type=str, help='Checkpoint path to use for evaluation.')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--n_repeats', type=int, default=10, help='Number of repeats for evaluation.')
    args = parser.parse_args()
    cfg_cli = {k: v for k, v in vars(args).items()}

    assert cfg_cli['model_type'] in ('miewid-msv2', 'miewid-msv3', 'MegaDescriptor-L-384', 'dinov2'), \
        f"Unknown model type: {cfg_cli['model_type']}"

    cfg = {}
    cfg['ks'] = [1, 3, 5, 7, 10, 20, 50]
    cfg['min_samples'] = 4
    cfg['distance_metric'] = 'cosine'  # cosine, euclidean etc
    cfg['weight_scheme'] = 'softmax'  # uniform or distance or softmax
    cfg['batch_size'] = 256
    cfg['num_workers'] = 8
    cfg['seed'] = 42
    cfg['n_repeats'] = 10
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Merge the CLI config into default config; the CLI arguments are prioritized
    for k, v in cfg_cli.items():
        cfg[k] = v

    if cfg['model_type'] == 'dinov2':
        cfg['opts'] = []
        cfg['crop_size'] = [224]
        cfg['input_size'] = [224]
        cfg['mean'] = [0.485, 0.456, 0.406]
        cfg['std'] = [0.229, 0.224, 0.225]
    elif cfg['model_type'] in ('miewid-msv2', 'miewid-msv3'):
        cfg['crop_size'] = [440]
        cfg['input_size'] = [440]
        cfg['model_tag'] = f'conservationxlabs/{cfg["model_type"]}'
        cfg['mean'] = [0.485, 0.456, 0.406]
        cfg['std'] = [0.229, 0.224, 0.225]
    elif cfg['model_type'] == 'MegaDescriptor-L-384':
        cfg['batch_size'] = 64
        cfg['crop_size'] = [384]
        cfg['input_size'] = [384]
        cfg['model_tag'] = 'hf-hub:BVRA/MegaDescriptor-L-384'
        cfg['mean'] = [0.5, 0.5, 0.5]
        cfg['std'] = [0.5, 0.5, 0.5]  # these are from the hugging-face card README and paper, but config.json says otherwise
        # cfg['mean'] = [0.485, 0.456, 0.406]
        # cfg['std'] = [0.229, 0.224, 0.225]

    run_nrepeats_on_dataset(cfg)
