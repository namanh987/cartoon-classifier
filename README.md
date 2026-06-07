# Cartoon Character Classifier

I built this to learn transfer learning properly — not just copy-paste a tutorial, but actually understand what's happening at each step. It classifies 5 cartoon characters using a pretrained ResNet-18 with a custom head trained from scratch.

Ended up at **86% test accuracy** with roughly 100 images per class, which I'm happy with given the dataset size.

---

## What it can recognise

Yellow Cartoon Characters: Homer Simpson, Jake the Dog (Adventure Time), Minion, Pikachu, SpongeBob.

The hardest pair for the model is Jake vs Pikachu — both yellow characters, and the confusion matrix shows it. 5 out of 30 Jakes got misclassified as Pikachu. Makes sense once you think about it.

---

## Dataset Design

Although the dataset is relatively small (around 100 images per class), I tried to prioritise **quality over quantity** when collecting and preparing the data.

A key goal was maintaining **class balance**, so each character is represented by a similar number of images. This helps reduce bias towards any particular class and makes the evaluation metrics more meaningful.

I also focused on **representativeness**. Rather than collecting many near-duplicate screenshots, I deliberately included images showing each character in different:

* Poses and body positions
* Facial expressions
* Viewing angles
* Backgrounds and environments
* Lighting conditions
* Image resolutions and artwork styles

This diversity encourages the model to learn the visual characteristics that define each character rather than memorising specific images or background patterns.

Since all five classes are yellow cartoon characters, creating a representative dataset was particularly important. The challenge is not recognising colour, but learning the differences in shape, facial features, proportions, clothing, and outlines that distinguish one character from another.

With a dataset of this size, careful data collection and augmentation had a significant impact on the model's ability to generalise to unseen images.


## Results

| | Value |
|---|---|
| Val accuracy | 94.63% |
| Test accuracy | 86.09% |
| Stopped at epoch | 25 (early stopping) |
| Training images | ~500 (~100 per class) |
| Test images | 151 |

Per-class breakdown:

| Character | F1 |
|---|---|
| Minion | 0.97 |
| SpongeBob | 0.88 |
| Homer | 0.84 |
| Pikachu | 0.84 |
| Jake | 0.77 |

---

## How it works

I didn't train ResNet-18 from scratch — that would need way more data. Instead I took the pretrained backbone (trained on 1.2M ImageNet images), froze all its weights, and only trained a small 2-layer classifier head on top. The backbone already knows how to see; I just taught it what to look for.

The head is: `Linear(512→128) → BatchNorm → ReLU → Dropout → Linear(128→5)`

Training uses early stopping so it quits automatically when validation loss stops improving, and a step LR scheduler that halves the learning rate every 20 epochs.

---

## Project structure

```
cartoon-classifier/
├── src/
│   ├── model.py          # model architecture
│   ├── train.py          # training loop
│   ├── test.py           # evaluation on test set
│   └── data_handling.py  # converts raw images to clean RGB JPEGs
├── inference.py          # run predictions + Grad-CAM from command line
├── data/                 # not in repo — see setup below
├── outputs/              # not in repo — created automatically when you train
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/cartoon-classifier
cd cartoon-classifier
pip install -r requirements.txt
```

Your data folder needs to look like this:

```
data/
├── train/
│   ├── homer/
│   ├── jake/
│   ├── minion/
│   ├── pikachu/
│   └── spongebob/
├── val/
│   └── ...
└── test/
    └── ...
```

Then run this once to normalise everything to RGB JPEG:

```bash
python src/data_handling.py
```

---

## Training

```bash
python src/train.py
```

Saves the best checkpoint to `outputs/best_model.pth` and plots training curves to `outputs/training_history.png`. Stops early if validation loss doesn't improve for 15 epochs.

If you want to tweak things:

```bash
python src/train.py --epochs 80 --lr 0.0005 --batch_size 32
```

---

## Evaluation

```bash
python src/test.py --checkpoint outputs/best_model.pth
```

Prints accuracy, per-class precision/recall/F1, and a confusion matrix. Also saves a random test sample with prediction probabilities to `outputs/test_sample.png`.

---

## Inference + Grad-CAM

```bash
# single image
python inference.py --checkpoint outputs/best_model.pth --image photo.jpg

# with Grad-CAM heatmap showing what the model focused on
python inference.py --checkpoint outputs/best_model.pth --image photo.jpg --gradcam

# run on a whole folder
python inference.py --checkpoint outputs/best_model.pth --folder test_images/ --gradcam --topk 3
```

Grad-CAM works by hooking into the last conv layer (`layer4`), computing how much each feature map contributed to the predicted class, and overlaying that as a heatmap. Green/yellow areas are what drove the decision.

---

## A few things I learned building this

Freezing the backbone isn't just about speed — with small datasets it's genuinely the right call. When I tried unfreezing everything the model memorised the training set almost immediately.

`model.state_dict()` returns references, not a copy. I had a bug early on where my "best" checkpoint kept getting overwritten because I wasn't using `copy.deepcopy`. Took me a while to figure out why the saved model always matched the final epoch.

`pin_memory=True` in the DataLoader causes a warning on CPU — it's only useful when transferring data to GPU. Small thing but worth knowing.

Data augmentation turned out to be more important than I expected. With only around 100 images per class, the model can run out of things to learn pretty quickly. Random flips, crops and colour variations helped it generalise much better instead of just memorising the training images.

I was honestly surprised by how well a pretrained ResNet-18 worked on such a small dataset. Transfer learning feels a bit like cheating sometimes — the network already knows a huge amount about visual features before it ever sees your images.

This project also reminded me that there's usually no need to reinvent the wheel. My first instinct was to build a custom CNN from scratch, but well-established architectures exist for a reason. A simple ResNet-18 gave strong results with very little tuning and saved me a lot of experimentation.

One thing I found particularly interesting is that every class in the dataset is yellow. At first I thought colour would dominate the predictions, but the model still learned to separate the characters surprisingly well. It wasn't looking for "yellow" — it was learning shapes, facial features, outlines, proportions and other distinctive visual patterns.

Finally, validation accuracy can look better than reality on a small dataset. The model reached over 94% validation accuracy, but test accuracy settled around 86%. Still a solid result, but a good reminder that the test set is what really matters.

---

## Troubleshooting

**DataLoader errors on Windows** — add `--num_workers 0` to your command.

**CUDA out of memory** — reduce `--batch_size`.

**`outputs/` folder created in the wrong place** — always run scripts from the project root, not from inside `src/`.