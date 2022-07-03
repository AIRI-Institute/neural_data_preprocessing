from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt

from .annotations import Annotation, Annotations

log = logging.getLogger(__name__)

T = TypeVar("T", bound=npt.NBitBase)
SignalArray = npt.NDArray[np.floating[T]]  # array of shape (n_samples, n_sensors)


@dataclass
class Signal(Generic[T]):
    """
    Timeseries stored as numpy array together with sampling rate and
    annotations

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_channels)
        Array with signal data
    sr : float
        Sampling rate
    annotations: Annotations
        Signal annotations

    """

    data: SignalArray[T]
    sr: float
    annotations: Annotations

    def __post_init__(self) -> None:
        assert self.data.ndim == 2, f"2-d array is required; got data of shape {self.data.shape}"

    def __len__(self) -> int:
        """Signal length in samples"""
        return self.n_samples

    @overload
    def __array__(self) -> SignalArray[T]:
        ...

    @overload
    def __array__(self, dtype: npt.DTypeLike) -> Any:
        ...

    def __array__(self, dtype: npt.DTypeLike | None = None) -> SignalArray[T] | npt.NDArray[Any]:
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data

    def __str__(self) -> str:
        return (
            f"signal of shape {self.n_samples} samples X {self.n_channels} channel(s) "
            + f" sampled at {self.sr:.2f} Hz;"
            + f" duration={round(self.duration, 2)} sec;"
            + f" {len(self.annotations)} annotated segments"
        )

    def update(self, data: SignalArray[T]) -> Signal[T]:
        assert len(data) == len(self), f"Can only update with equal length data; got {len(data)=}"
        return self.__class__(data, self.sr, self.annotations)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def duration(self) -> float:
        """Signal duration in seconds"""
        return self.n_samples / self.sr


class Signal1D(Signal[T], Generic[T]):
    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.n_channels == 1


def split_into_good_segments(signal: Signal[T], round_annot_times: bool = True) -> list[Signal[T]]:
    good_signal = drop_bad_segments(signal, round_annot_times)
    res: list[Signal[T]] = []
    annots_segment: Annotations = []
    offset = 0.0
    segment_start_samp = 0
    for a in good_signal.annotations:
        if a.type == "BAD_BOUNDARY":
            this_samp = int(a.onset * good_signal.sr)
            if segment_start_samp == this_samp:
                continue
            data_segment = good_signal.data[segment_start_samp:this_samp]
            res.append(Signal(data_segment, good_signal.sr, annots_segment))
            segment_start_samp = this_samp
            offset += len(data_segment) / good_signal.sr
            annots_segment = []
        else:
            onset = round(a.onset - offset, 4) if round_annot_times else a.onset - offset
            duration = round(a.duration, 4) if round_annot_times else a.duration
            annots_segment.append(Annotation(onset, duration, a.type))
    this_samp = len(good_signal)
    if segment_start_samp != this_samp:
        data_segment = good_signal.data[segment_start_samp:this_samp]
        res.append(Signal(data_segment, good_signal.sr, annots_segment))
    return res


def drop_bad_segments(signal: Signal[T], round_annot_times: bool = True) -> Signal[T]:
    log.debug(f"Drop bad segments: received signal: {str(signal)}")
    normalized_annots = _AnnotationsOverlapProcessor(signal.annotations).normalize_bad_segments()
    removed_dur = 0.0
    new_annots: Annotations = []
    good_segments_mask = np.ones(len(signal), dtype=bool)

    for a in normalized_annots:
        onset = round(a.onset - removed_dur, 4) if round_annot_times else a.onset - removed_dur
        duration = round(a.duration, 4) if round_annot_times else a.duration
        if not a.type.startswith("BAD"):
            new_annots.append(Annotation(onset, duration, a.type))
        else:
            segment_start_samp = int(a.onset * signal.sr)
            segment_end_samp = int((a.onset + a.duration) * signal.sr)
            good_segments_mask[segment_start_samp:segment_end_samp] = False
            new_annots.append(Annotation(onset=onset, duration=0, type="BAD_BOUNDARY"))
            removed_dur += a.duration

    return Signal(signal.data[good_segments_mask], signal.sr, new_annots)


class _AnnotationsOverlapProcessor:
    # order important: when segment start and end occur simoultaneosly, first
    # add new bad segment, then remove old
    START = 0
    END = 1

    def __init__(self, annotations: Annotations, round_annot_times: bool = True):
        self.events = []
        for a in annotations:
            self.events.append((a.onset, self.START, a))
            self.events.append((a.onset + a.duration, self.END, a))
        self.events.sort()
        self.bad_overlap_cnt = 0
        self.res: Annotations = []
        self.open_segments: dict[Annotation, float] = {}
        self.bad_start: float | None = None
        self._process_all()
        if round_annot_times:
            self._round_results()

    def normalize_bad_segments(self) -> Annotations:
        """
        Make bad segments disjoint; make sure the good don't overlap them.

        Merge intersecting bad segments; for good segments crop parts
        overlaping with the bad ones.

        """
        return self.res

    def _round_results(self, n=4):
        self.res = [Annotation(round(a.onset, n), round(a.duration, n), a.type) for a in self.res]

    def _process_all(self) -> None:
        for ev_onset, ev_type, ev_annotation in self.events:
            self._process_event(ev_onset, ev_type, ev_annotation)
        assert not self.open_segments

    def _process_event(self, ev_onset: float, ev_type, ev_annotation: Annotation) -> None:
        is_bad = ev_annotation.type.startswith("BAD")
        if ev_type == self.START and is_bad:
            if self.bad_overlap_cnt == 0:
                self._finalize_all_open_segments(end=ev_onset)
                self.bad_start = ev_onset
            self.bad_overlap_cnt += 1
        elif ev_type == self.START and not is_bad:
            self._start_tracking_good_segment(ev_annotation)
        elif ev_type == self.END and is_bad:
            self.bad_overlap_cnt -= 1
            if self.bad_overlap_cnt == 0:
                self._change_tracked_segments_onset(new_onset=ev_onset)
                assert self.bad_start is not None
                self.res.append(Annotation(self.bad_start, ev_onset - self.bad_start, "BAD"))
        else:  # good segment end
            self._stop_tracking_good_segment(ev_onset, ev_annotation)

    def _start_tracking_good_segment(self, annotation):
        self.open_segments[annotation] = annotation.onset

    def _change_tracked_segments_onset(self, new_onset):
        self.open_segments.update((a, new_onset) for a in self.open_segments)

    def _finalize_all_open_segments(self, end):
        self.res.extend(Annotation(o, end - o, a.type) for a, o in self.open_segments.items())

    def _stop_tracking_good_segment(self, ev_onset, annotation):
        annot_onset = self.open_segments.pop(annotation)
        if self.bad_overlap_cnt == 0:
            self._finalize_segment(ev_onset, annot_onset, annotation)

    def _finalize_segment(self, ev_onset, annot_onset, annot):
        self.res.append(Annotation(annot_onset, ev_onset - annot_onset, annot.type))
