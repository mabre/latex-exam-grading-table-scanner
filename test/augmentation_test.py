import numpy as np

from training.augmentation import scale_image_with_border


def test_scale_image_with_border_smaller():
    image = np.array([
        [200, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 200]], dtype=np.uint8)
    scaled = scale_image_with_border(image, target_size=6, min_scale=0.66, max_scale=0.66)
    assert scaled.shape == (6, 6)
    assert scaled[0][0] == 255
    assert 100 <= scaled[2][1] <= 255
    assert scaled[2][2] == 100


def test_scale_image_with_border_bigger():
    image = np.array([
        [200, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 200]], dtype=np.uint8)
    scaled = scale_image_with_border(image, target_size=6, min_scale=1.5, max_scale=1.5)
    assert scaled.shape == (6, 6)
    assert 100 <= scaled[0][0] <= 200
    assert scaled[2][2] == 100