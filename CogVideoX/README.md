
493b:
DiffusionAsShader: used for processing data
DAS_training: clean training directory

test_model
DAS_50:
493b: cnet_4600


Input shape
```python
# tracks: shape -> 1, 49, n, 3
# visibilities: shape -> 1, 49, n
track_dict = {"tracks": tracks, "visibility": visibilities}
np.save(track_path, track_dict)

masks = np.ones((1, 480, 720), dtype=bool)
mask2rle(masks, rle_path)
```