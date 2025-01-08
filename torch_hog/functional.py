from collections.abc import Iterable
import torch
from einops import rearrange, repeat
from kornia.color import rgb_to_grayscale
from kornia.filters import spatial_gradient
from kornia.utils import create_meshgrid
from torch.nn import functional as F
from torchvision.ops import roi_align


__all__ = ["hog"]


def _batch_histogram(input, bins, weight=None):
    """
    Compute histograms for a batch of tensors.

    Args:
        input (torch.Tensor): The input tensor of shape (B, D).
        bins (torch.Tensor): The bins tensor of shape (K,). Defines K-1 bins.
        weight (torch.Tensor): The weight tensor of shape (B, D).

    Returns:
        torch.Tensor: The histogram tensor of shape (B, K-1).
    """
    B, D = input.shape
    K = bins.shape[0]

    # verify bins are sorted
    if not (bins[:-1] < bins[1:]).all():
        raise ValueError("Bins must be sorted in ascending order")

    buckets = torch.bucketize(input, bins, right=True)  # (B, D) in the range of [0, K]

    one_hot = F.one_hot(buckets, num_classes=K + 1)  # (B, D, K+1)
    one_hot = one_hot[:, :, 1:-1]  # remove the first and last bin as they are out of range. # (B, D, K-1)

    if weight is None:
        weight = torch.ones_like(input)

    weight = repeat(weight, "b d -> b d k", k=K - 1)  # (B, D, K-1)
    histogram = (one_hot * weight).sum(dim=1)  # (B, K-1)

    return histogram


def hog(img, coords=None, num_bins=9, cell_size=8, padding="reflect"):
    device = img.device

    # convert img to (B, C, H, W) if not already
    if img.dim() == 2:
        img = rearrange(img, "h w -> 1 1 h w")
    elif img.dim() == 3:
        img = rearrange(img, "c h w -> 1 c h w")

    assert img.dim() == 4, f"Input tensor must be 2D or 3D, got {img.dim()} dimensions of shape {img.shape}"

    # convert to grayscale if not already
    if img.shape[1] > 1:
        img = rgb_to_grayscale(img)

    B, C, H, W = img.shape

    assert C == 1, f"Input tensor must have 1 channel, got {img.shape}"

    if coords is None:
        # generate grid coordinates in the range [0, H) x [0, W)
        coords = create_meshgrid(H, W, normalized_coordinates=False, device=device)
        coords = rearrange(coords, "1 h w d -> (h w) d", d=2)  # (N, 2)
    elif isinstance(coords, Iterable):
        coords = list(coords)

    if not (isinstance(coords, list) or isinstance(coords, torch.Tensor)):
        raise ValueError("Coordinates must be a list of tensors or a tensor of shape (N, 2)")

    # convert to list if not given as a list, and the user wants to use the same coordinates for all images
    broadcast = False
    if isinstance(coords, torch.Tensor) and coords.dim() == 2 and coords.shape[1] == 2:
        coords = [coords] * B
        broadcast = True

    # verify the coordinates are a list of tensors with shape (N, 2)
    for coord in coords:
        assert isinstance(coord, torch.Tensor), f"Coordinates must be a list of tensors, got {type(coord)}"
        assert (
            coord.dim() == 2 and coord.shape[1] == 2
        ), f"Coordinates must be a tensor with shape (N, 2), got {coord.shape}"

    # pad the images to make sure coordinates around the borders will have valid patches
    pad = cell_size
    img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode=padding)

    for i in range(len(coords)):
        coords[i] = (coords[i] + pad).float()

    # extract patches around the coordinates. add +1 and -1 to include the borders gradients
    half_cell = (cell_size + 1) // 2  # round up
    boxes = [
        torch.stack(
            [
                coord[:, 0] - half_cell - 1,
                coord[:, 1] - half_cell - 1,
                coord[:, 0] + half_cell + 1,
                coord[:, 1] + half_cell + 1,
            ],
            dim=1,
        )
        for i, coord in enumerate(coords)
    ]  # list of (N, 4) boxes

    patches = roi_align(img, boxes, output_size=cell_size + 2, aligned=True)  # (K, C, cell_size+2, cell_size+2)

    grad = spatial_gradient(patches, mode="diff", normalized=False)  # (K, C, 2, cell_size+2, cell_size+2)
    grad = grad[:, 0, :, 1:-1, 1:-1]  # remove gradient padding and channel dimension

    magnitude = grad.norm(dim=1)  # (K, cell_size, cell_size)
    orientation = (grad[:, 1] / grad[:, 0]).atan().rad2deg()  # (K, cell_size, cell_size) in the range [-90, 90]

    # project orientation to [0, 180]
    orientation = (orientation + 180) % 180

    magnitude = rearrange(magnitude, "k h w -> k (h w)")
    orientation = rearrange(orientation, "k h w -> k (h w)")

    bin_tensor = torch.arange(0, 180 + 1, 180 // num_bins).to(device)  # (num_bins+1,)
    hog = _batch_histogram(orientation, bins=bin_tensor, weight=magnitude)  # (K, num_bins)

    # split the hog tensor into a list of tensors according to the number of coordinates
    splits = torch.tensor([len(coord) for coord in coords], device="cpu").cumsum(0)
    hogs = torch.tensor_split(hog, splits)[:-1]  # last element is empty - remove it

    # if we broadcasted the coordinates, stack the hogs into a single tensor, as the user expects
    if broadcast:
        hogs = torch.stack(hogs, dim=0) # (B, N, num_bins)

    return hogs
