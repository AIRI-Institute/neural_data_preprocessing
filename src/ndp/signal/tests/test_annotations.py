import numpy as np
import numpy.testing as nptest

from ndp.signal.annotations import Annotation, annots_from_mask


def test_annotation_as_mask_returns_correct_mask_for_nonzero_duration_segment():
    a = Annotation(1, 1, "test")
    m = a.as_mask(nsamp=10, sr=2)
    assert len(m) == 10
    nptest.assert_array_equal(
        m, np.array([False, False, True, True, True, False, False, False, False, False])
    )


def test_annotation_as_mask_returns_correct_mask_for_zero_duration_segment():
    a = Annotation(1, 0, "test")
    m = a.as_mask(nsamp=10, sr=2)
    assert len(m) == 10
    nptest.assert_array_equal(
        m, np.array([False, False, True, False, False, False, False, False, False, False])
    )


def test_annots_from_mask_convert_back_to_the_same_annotations():
    annots = [Annotation(1, 1, "test"), Annotation(3, 0, "test")]
    masks = [a.as_mask(nsamp=10, sr=2) for a in annots]

    mask = np.logical_or.reduce(masks)
    assert annots_from_mask(mask, sr=2, type="test") == annots
