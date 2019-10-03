# PyTorch-MixNet-S
Extremely light-weight MixNet with ImageNet Top-1 75.7% accuracy and 2.5M parameters.

## [Model Link](./mixnet-ss.pth)

## Val setting
```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC), # == 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
```
