from __future__ import annotations

import logging
from typing import Optional

import librosa as lb  # type: ignore
import librosa.feature as lbf  # type: ignore
import numpy as np
import numpy.typing as npt
import scipy.signal as scs  # type: ignore

from . import Signal, Signal1D, T

log = logging.getLogger(__name__)


def asd(signal: Signal, n: Optional[int] = None) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute amplitude spectral density

    Parameters
    ----------
    signal: Signal
        Input signal
    n: int, optional
        Signal length in samples; if doesn't equal len(signal), the signal
        is cropped or padded to match n (see numpy.fft.fft)

    Returns
    -------
    freqs: array
        1D array of frequencies
    amp_spec: array
        Amplitude spectrum

    See also
    --------
    numpy.fft.fft
    numpy.fft.fftfreq

    """
    apm_spec = np.abs(np.fft.fft(signal.data, n, axis=0))
    n = signal.n_samples if n is None else n
    freqs = np.fft.fftfreq(n, 1 / signal.sr)
    end = len(freqs) // 2
    assert n // 2 == end, f"{n=}, {end=}, {signal.n_samples=}, {freqs=}"
    return freqs[:end], apm_spec[:end, :]


def mfccs(signal: Signal, d: int, out_nsamp: int, n_mfcc: int) -> npt.NDArray:
    m = lbf.mfcc(y=signal, sr=int(signal.sr), hop_length=d, n_mfcc=n_mfcc)
    mfccs_resampled: npt.NDArray = scs.resample(x=m.T, num=out_nsamp)
    return mfccs_resampled


def logmelspec(signal: Signal1D[T], n: int, f_max: float, d: int) -> Signal[T]:
    melspec = lbf.melspectrogram(
        y=np.squeeze(np.asarray(signal)),
        sr=int(signal.sr),
        n_mels=n,
        fmax=f_max,
        hop_length=d,
        n_fft=1024,
    )
    log.debug(f"{melspec.shape=}")
    return Signal(
        lb.power_to_db(melspec, ref=np.max).astype(signal.dtype).T,  # pyright: ignore
        signal.sr / d,
        signal.annotations,
    )


# def classic_lpc_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, order):
#     WIN_LENGTH = 1001
#     sound /= np.max(np.abs(sound))
#     lpcs = sp.extract_lpcs(sound, order, WIN_LENGTH, downsampling_coef, ecog_size)
#     lpcs = skp.scale(lpcs)
#     return lpcs


# def classic_mfcc_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc):
#     sound /= np.max(np.abs(sound))
#     mfccs = sp.extract_mfccs(sound, sampling_rate, downsampling_coef, ecog_size, n_mfcc)
#     mfccs = skp.scale(mfccs)
#     return mfccs
# def lpcs(self, order: int, window: int, d: int, out_nsamp: int) -> Optional[npt.NDArray]:
#     w2 = window // 2
#     pad: npt.NDArray = np.pad(self.signal, (w2, w2))
#     r = [lb.lpc(pad[i - w2 : i + w2 + 1], order) for i in range(w2, len(pad) - w2 + 1, d)]
#     res: npt.NDArray = np.array(r)[:, 1:]
#     # todo: remove this terrible ifs
#     if res.shape[0] == out_nsamp:
#         return res
#     elif res.shape[0] - 1 == out_nsamp:
#         return res[:-1]
#     else:
#         raise ValueError
