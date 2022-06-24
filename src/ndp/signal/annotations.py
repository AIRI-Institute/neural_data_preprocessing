from __future__ import annotations

from typing import List, NamedTuple

import numpy as np
import numpy.typing as npt

SignalMaskArray = npt.NDArray[np.bool_]  # array of shape (n_samples,)


class Annotation(NamedTuple):
    onset: float
    duration: float
    type: str

    def as_mask(self, sr: float, nsamp: int) -> SignalMaskArray:
        res = np.zeros(nsamp, dtype=np.bool_)
        start = round(self.onset * sr)
        end = round((self.onset + self.duration) * sr) + 1
        res[start:end] = True
        return res


Annotations = List[Annotation]


def annots_from_mask(mask: SignalMaskArray, sr: float, type: str) -> Annotations:
    res: Annotations = []
    if not len(mask):
        return res
    prev_seg_start = 0 if mask[0] else None
    prev_sample = mask[0]
    for i, sample in enumerate(mask[1:], start=1):
        if not prev_sample and sample:
            prev_seg_start = i
        elif prev_sample and not sample:
            # one sample segment counts as zero-length
            assert prev_seg_start is not None
            res.append(Annotation(prev_seg_start / sr, (i - 1 - prev_seg_start) / sr, type))
            prev_seg_start = None
        prev_sample = sample
    if prev_seg_start is not None:
        res.append(Annotation(prev_seg_start / sr, (len(mask) - 1 - prev_seg_start) / sr, type))
    return res
