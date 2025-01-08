# torch-hog
Non Official PyTorch Implementation of the HoG Descriptor

## Recent updates:
- 0.0.1: Basic HoG calcuation was added

## Contents

- [Installation](#Installation)
- [Basic Use](#usage)
- [Citing](#Citing)
- [Repository](https://github.com/wamitw/torch-hog) and [discussions](https://github.com/wamitw/torch-hog/discussions)

## Installation  <a name="Installation"></a>

Plain and simple:
```bash
pip install torch-hog
```


## Basice Use  <a name="usage"></a>
### Import
Use as a PyTorch Module:
```python
from torch_hog import HoG

hog = HoG(num_bins=9, cell_size=8, padding="reflect")
```

Or as a function:
```python
from torch_hog.functional import hog
```

### Dense Feature Extraction
either way, the API is the same. To get the features at every possible coordinate, just feed in the batch:
```python
# Create a random batch of images
B, C, H, W = 5, 3, 360, 480
img = torch.randn((B, C, H, W))

dense_hog_features = hog(img) # returns a tensor of shape (B, H*W, bins)
```

### Sparse Feature Extraction
You can specify the desired coordinates for every image in the batch.
```python
# specify the desired coordinates
coords = torch.tensor([
  [20, 15],
  [17.7, 18.9], # yes we're accepting floats too
  [150, 170],
])

# Create a random batch of images
B, C, H, W = 5, 3, 360, 480
img = torch.randn((B, C, H, W))

hog_features = hog(img, coords=coords) # returns a tensor of shape (B, 3, bins)
```


You can also specify different number of coordinates for every image:
```python
# specify the desired coordinates
coords1 = torch.tensor([
  [20, 15],
  [17.7, 18.9], # yes we're accepting floats too
  [150, 170],
])

coords2 = torch.tensor([
  [41, 38],
  [69, 85],
])
coords = [coords1, coords2]

# Create a random batch of images
B, C, H, W = 5, 3, 360, 480
img = torch.randn((B, C, H, W))

hog_features = hog(img, coords=coords) # returns a list of tensors as with the shapes: [(3, bins), (2, bins)]
```

## Cite Us <a name="Citing"></a>

Please use the following bibtex record if you're using this project in your research:

```text
@misc{TorchHog2025,
  author = {Abe, Amit},
  title = {Torch-HoG},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wamitw/torch-hog}},
}
```
