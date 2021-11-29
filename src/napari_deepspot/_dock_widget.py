"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""

from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout


class EnhanceSpot(QWidget):
    def __init__(self, napari_viewer):

        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.model_path = "./src/models/MHybrid/"

        btn = QPushButton("Enhance")
        btn.clicked.connect(self._on_click)
        self.layout().addWidget(btn)


    def _on_click(self):
        import napari_deepspot.deepSpot_functions as df
        image = df.prepare_image(self)
        if image is not None:
            image = df.enhance(self, image)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return EnhanceSpot