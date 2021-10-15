
## ImageNet-100

```
python imagenet_100_subset.py --imagenet_path /path/to/imagenet/ --imagenet100_path /path/to/imagenet-100/
```

## ImageNet-Texture

![image](https://user-images.githubusercontent.com/22885450/137448411-79e98f26-4d74-4bc1-a4cc-908899f11257.png)


First install the texture-synthesis tool from https://github.com/EmbarkStudios/texture-synthesis. Then run the following script. Note that this could take hours to generate the entire dataset. You can further accelerate with multi-processing.

``
python imagenet_100_subset.py --imagenet_path /path/to/imagenet/ --imagenet100_path /path/to/imagenet-100/
``
