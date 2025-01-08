from torch.nn import Module

from .functional import hog


__all__ = ["HoG"]


class HoG(Module):
    def __init__(self, num_bins=9, cell_size=8, padding="reflect", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.cell_size = cell_size
        self.padding = padding

    def forward(self, img, coords=None):
        hogs = hog(img, coords=coords, num_bins=self.num_bins, cell_size=self.cell_size, padding=self.padding)

        return hogs
