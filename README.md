# PyTorch-MixNet-SS
Extremely light-weight MixNet with ImageNet Top-1 **75.7%** accuracy and **2.5M** parameters.

Precision | Top-1 (%) | Top-5 (%) | Params
------------ | ------------- | ----------- | ------------
FP32 | 75.744 | 92.576 | 2.5 M
FP16 | 75.714 | 92.570 | 1.3 M

### 1. [Download Link](./mixnet-ss.pth)

### 2. How to load params ?
```python
from collections import OrderedDict

state_dict = torch.load(args.pretrained)
new_state_dict = OrderedDict()
for key_ori, key_pre in zip(model.state_dict().keys(), state_dict.keys()):
    new_state_dict[key_ori] = state_dict[key_pre]
model.load_state_dict(new_state_dict)       
```

### 3. Inference setting :
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
