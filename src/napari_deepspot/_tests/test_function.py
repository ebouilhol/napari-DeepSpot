import napari_deepspot
import pytest


def test_custom_loss():
    import napari_deepspot.deepSpot_functions as dsf
    import numpy as np
    x = np.zeros((10,10,2))
    y = np.zeros((10,10,2))
    res = dsf.custom_loss(x,y)
    assert res==0
