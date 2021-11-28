"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory
import numpy as np
import scipy.ndimage as ndimage
import skimage.transform as transform
from napari_plugin_engine import napari_hook_implementation
from qtpy import QtCore
from qtpy.QtWidgets import QWidget, QPushButton, QCheckBox, QLabel, QVBoxLayout


class EnhanceSpot(QWidget):
    def __init__(self, napari_viewer):
        import deepmeta.deepmeta_functions as df
        super().__init__()
        self.cfg = df.load_config()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        btn = QPushButton("Enhance")
        btn.clicked.connect(self._on_click)
        self.layout().addWidget(btn)


    def _on_click(self):
        import deepSpot_functions as df
        image = df.load_img(self)
        if image is not None:
            self.viewer.add_image(image, name="spots")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [EnhanceSpot]