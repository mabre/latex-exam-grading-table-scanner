from collections import Counter

import numpy as np
import pytest

from training.train_model import merge_balanced


def test_merge_balanced() -> None:
    images_real = np.array([1, 2, 3, 4, 5])
    labels_real = np.array([0, 0, 0, 1, 2])
    images_augmented = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    labels_augmented = np.array([0, 0, 0, 1,  1,  1,  2,  2,  2,  3,  3,  3])
    images, labels = merge_balanced(images_real, labels_real, images_augmented, labels_augmented)

    assert len(images) == len(labels)
    assert len(images) == 12
    assert Counter(labels) == {0: 3, 1: 3, 2: 3, 3: 3}
    assert 6 not in set(images)
    assert 7 not in set(images)
    assert 8 not in set(images)


def test_merge_balanced_throws_when_too_few_samples() -> None:
    images_real = np.array([1, 2, 3, 4, 5])
    labels_real = np.array([0, 0, 0, 0, 2])
    images_augmented = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    labels_augmented = np.array([0, 0, 0, 1,  1,  1,  2,  2,  2,  3,  3,  3])
    with pytest.raises(ValueError, match="samples too few"):
        merge_balanced(images_real, labels_real, images_augmented, labels_augmented)


def test_merge_balanced_works_with_empty_real_data() -> None:
    images_real = np.array([])
    labels_real = np.array([])
    images_augmented = np.array([6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    labels_augmented = np.array([0, 0, 1,  1,  1,  2,  2,  2,  3,  3,  3])
    images, labels = merge_balanced(images_real, labels_real, images_augmented, labels_augmented)
    assert len(images) == 8