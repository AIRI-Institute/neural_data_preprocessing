from ndp.signal.annotations import Annotation


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


