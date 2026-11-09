"""Train-window clustering condition; not a reproduction of K-ProtoDiff."""
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre

BINS = [('0-30', 0, 30), ('31-60', 31, 60), ('61-90', 61, 90),
        ('91-124', 91, 124), ('125', 125, 125)]


def features(windows):
    q = windows.shape[1] // 4
    return np.concatenate([windows.mean(1), windows.std(1),
                           windows[:, -q:].mean(1)-windows[:, :q].mean(1)], axis=1)


def fit_prototypes(windows, rul, seed=20261103):
    scaler = StandardScaler().fit(features(windows))
    z = scaler.transform(features(windows))
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(z)
    ids = km.labels_.astype(np.int64)
    global_frequency = np.bincount(ids, minlength=4) / len(ids)
    frequency, empty = [], []
    for name, lo, hi in BINS:
        mask = (rul.reshape(-1) >= lo) & (rul.reshape(-1) <= hi)
        empty.append(not mask.any())
        frequency.append(np.bincount(ids[mask], minlength=4)/mask.sum()
                         if mask.any() else global_frequency)
    return dict(feature_mean=scaler.mean_, feature_scale=scaler.scale_,
                centers=km.cluster_centers_, prototype_id=ids,
                frequency=np.array(frequency), global_frequency=global_frequency,
                empty_bins=np.array(empty), cluster_seed=np.array(seed))


def draw_prototypes(rul, stats, rng):
    y = np.asarray(rul).reshape(-1)
    if not np.all((y >= 0) & (y <= 125) & (y == np.floor(y))):
        raise ValueError('Expected capped RUL in original integer cycles')
    ids = np.empty(len(y), dtype=np.int64)
    for i, (_, lo, hi) in enumerate(BINS):
        mask = (y >= lo) & (y <= hi)
        ids[mask] = rng.choice(4, mask.sum(), p=stats['frequency'][i])
    return ids


class PrototypeUNet(UNet1D_fre):
    def __init__(self, *args, prototype_seed=991, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve base initialization and subsequent global RNG draws in A/B.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(prototype_seed)
            dim = self.classes_emb.out_features
            self.prototype_embedding = nn.Embedding(4, dim)
            self.prototype_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.prototype_ids = None
        self.classes_emb.register_forward_hook(self._add_prototype)

    def _add_prototype(self, module, inputs, output):
        if self.prototype_ids is None:
            raise RuntimeError('Set prototype_ids once per training/sampling batch')
        return output + self.prototype_projection(self.prototype_embedding(self.prototype_ids))


class SamplingConditionMixin:
    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        # eval() alone does not disable the original random condition dropout.
        conditional = self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1:
            return conditional
        # Original shared dropout mask replaces the combined embedding with
        # null_classes_emb, hence the unconditional branch has no prototype.
        null = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null + (conditional-null)*cond_scale


class SamplingBase(SamplingConditionMixin, UNet1D_fre):
    pass


class SamplingPrototype(SamplingConditionMixin, PrototypeUNet):
    pass


def make_model(enabled=False, sampling=False):
    cls = (SamplingPrototype if enabled else SamplingBase) if sampling else (PrototypeUNet if enabled else UNet1D_fre)
    return cls(dim=32, dim_mults=(1, 2), cond_drop_prob=.2, channels=14, length=48)
