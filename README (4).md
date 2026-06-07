# Models folder

Place your trained checkpoint here:

```
models/resnet18_cifar10.pth
```

It should be a PyTorch checkpoint produced by the training script in the
[`deep-learning-tasks`](../../deep-learning-tasks/task1_image_classifier/) repo,
i.e. a dict with at least:

```python
{
    "model_state_dict": <state_dict>,
    "classes": ["airplane", "automobile", ..., "truck"],  # optional
}
```

> If no checkpoint is present, the API still starts using a freshly initialized
> ResNet18 head (predictions will be random until you provide a trained file).

The API looks for the file at the path defined by the `MODEL_PATH` env var,
defaulting to `models/resnet18_cifar10.pth`.

If your model is too big for GitHub, upload it to **Google Drive** and link it
here.
