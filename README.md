# Universal Chimpanzee Face Embedder

```bibtex
@InProceedings{self2025iashin,
  title={Self-supervised Learning on Camera Trap Footage Yields a Strong Universal Face Embedder},
  author={Iashin, V., Lee, H., Schofield, D., and Zisserman, A.},
  booktitle={2025 IEEE International Conference on Image Processing Challenges and Workshops (ICIPCW)},
  year={2025},
  organization={IEEE}
}
```

[Project Page](https://www.robots.ox.ac.uk/~vgg/research/ChimpUFE/) • [Paper](https://arxiv.org/abs/2507.10552)

This repo provides two main functionalities at the moment:
1. **Face Recognition**: Compare a query image to a gallery of images using a Universal Face Embedder model to find similar faces or compute similarity scores.
2. **Camera Trap Detection**: Detect and track animals (chimpanzees) in video footage using YOLOX object detection with ByteTrack for multi-object tracking.
3. **Evaluation on PetFace-Chimp**: Benchmark re-ID and verification: builds KNN over embeddings and computes ROC-AUC.

## Installation
Install the required packages in your python environment using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Files and Folders
- `demo_face_rec.py`: Main script for face recognition inference.
- `demo_camera_trap.py`: Main script for camera trap detection and tracking.
- `assets/`: Example images, videos, and gallery structure.
- `src/face_embedder/`: Source code for the Universal Chimpanzee Face Embedder.
- `src/tracker/`: Source code for YOLOX object detection and  tracking.

## Pre-trained Weights

**Latest — v1.1 (2025-09)**
- Download: [25-08-29T11-49-28_340k.pth](https://github.com/v-iashin/ChimpUFE/releases/download/v1.1/25-08-29T11-49-28_340k.pth)
- Summary: Improves PetFace-Chimp re-id accuracy by **+1.4%** and verification ROC-AUC by **+0.5%** vs v1.0.

**Previous — v1.0 (2025-07)**
- Download: [25-06-06T20-51-36_330k.pth](https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/25-06-06T20-51-36_330k.pth)


Before running the demos, download the pre-trained weights for the ViT-Base version from [here (415 MB)](https://github.com/v-iashin/ChimpUFE/releases/download/v1.1/25-08-29T11-49-28_340k.pth) and place them in the `./assets/weights` folder:
```bash
# face recognition weights
wget -P ./assets/weights https://github.com/v-iashin/ChimpUFE/releases/download/v1.1/25-08-29T11-49-28_340k.pth
```
and if you'd like to run the camera trap demo, download the YOLOX weights from [here (378 MB)](https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/yolox_best_only_model.pth)
```bash
# camera trap weights
wget -P ./assets/weights https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/yolox_best_only_model.pth
```

**Note**: All pre-trained weights are licensed under non-commercial use only.

## Usage

### 1. Compare Two Images Directly
This prints the similarity score between two images from [PetFace dataset](https://dahlian00.github.io/PetFacePage/):
```bash
# different individuals
python demo_face_rec.py \
  --pretrained_weights ./assets/weights/25-08-29T11-49-28_340k.pth \
  --query_path ./assets/gallery/000000/01.png \
  --gallery_path ./assets/gallery/000001/00.png

# Cosine similarity between the query and the gallery image: 0.1481

# same individual
python demo_face_rec.py \
  --pretrained_weights ./assets/weights/25-08-29T11-49-28_340k.pth \
  --query_path ./assets/gallery/000000/01.png \
  --gallery_path ./assets/gallery/000000/02.png

# Cosine similarity between the query and the gallery image: 0.4365
```

**Picking a Threshold for Face Recognition.**
Verification of face recognition results often requires setting a threshold on the (cosine) similarity scores.
This threshold determines whether a pair of faces is considered a match or not.
You can choose this threshold based on the desired trade-off between true positive rate (TPR) and false positive rate (FPR).
Here are some example thresholds based on the ROC curve analysis on PetFace-Chimp dataset:

```
TPR @ 0.1% FPR = 9.14% (threshold=0.57912)
TPR @ 1.0% FPR = 20.72% (threshold=0.49020)
TPR @ 5.0% FPR = 36.95% (threshold=0.41365)
TPR @ 10.0% FPR = 47.07% (threshold=0.37319)
```

Thus, for a given false positive rate (FPR), i.e. if we allow 10% of the pairs to be falsely identified as matches,
we can expect to correctly identify 47.07% of the true matches (true positive rate, TPR) with a threshold of `0.37319`.

*Note*: The threshold values are dataset-specific, so you may need to adjust them based on your dataset and requirements.

### 2. Compare a Query Image to a Gallery Folder
This finds the top-k most similar images in the gallery to your query image.
The gallery folder should be structured as subfolders (e.g., `gallery/000000/`, `gallery/000001/`, etc.) each containing images.
We are going to use a few examples from the [PetFace dataset](https://dahlian00.github.io/PetFacePage/) for demonstration purposes.

```bash
python demo_face_rec.py \
  --pretrained_weights ./assets/weights/25-08-29T11-49-28_340k.pth \
  --query_path ./assets/gallery/000003/00.png \
  --gallery_path ./assets/gallery \
  --top_k 10

# Top 10 matches for query './assets/gallery/000003/00.png':
#  1. ./assets/gallery/000003/00.png | similarity: 1.0000
#  2. ./assets/gallery/000003/01.png | similarity: 0.6383
#  3. ./assets/gallery/000003/03.png | similarity: 0.5392
#  4. ./assets/gallery/000003/02.png | similarity: 0.4892
#  5. ./assets/gallery/000002/00.png | similarity: 0.4883
#  6. ./assets/gallery/000003/04.png | similarity: 0.4850
#  7. ./assets/gallery/000002/02.png | similarity: 0.4385
#  8. ./assets/gallery/000002/03.png | similarity: 0.4204
#  9. ./assets/gallery/000002/04.png | similarity: 0.3939
# 10. ./assets/gallery/000002/01.png | similarity: 0.3713
```

### 3. Camera Trap Demo
This runs the camera trap demo, which detects and tracks animals in a video (e.g. `./assets/loma_mt.mp4` obtained from [YouTube](https://www.youtube.com/watch?v=9f_KBEQOspU)).

```bash
python demo_camera_trap.py \
  --video_path ./assets/loma_mt.mp4 \
  --pretrained_weights ./assets/weights/yolox_best_only_model.pth \
  --save_results_path ./assets/detections \
  --fp16 --fuse

# Tracking results saved to ./assets/detections/loma_mt.txt
# Columns: frame_id, track_id, left, top, width, height, score, -1, -1, -1
# 0,1,594.00,255.60,110.70,117.45,0.95,-1,-1,-1
# 0,2,179.77,248.17,26.10,49.50,0.87,-1,-1,-1
# 1,1,604.48,254.04,109.26,115.89,0.95,-1,-1,-1
# 1,2,179.77,248.17,26.10,49.50,0.87,-1,-1,-1
# 2,1,616.73,252.01,109.26,115.62,0.95,-1,-1,-1
# 2,2,179.95,248.17,25.92,49.15,0.87,-1,-1,-1
# ...
```


The results will be saved in the `assets/detections` folder.

If you want to overlay the tracking results on the video, you can add the `--overlay_results_on_video` flag
and find the results in `./assets/detections/vis`.


### 4. Evaluation on PetFace-Chimp

We provide a script to benchmark our model on the chimpanzee subset of the PetFace dataset (**PetFace-Chimp**) for both re-identification (re-ID) and verification. We remove near-duplicates, discard faulty images (~12%), and drop classes with fewer than 4 usable portraits, leaving **2,853 images across 376 IDs**. The discarded files are listed in `./assets/PetFace/filtered_petface_files.txt`.

#### Prepare the data
First, obtain the original `chimp.tar.gz` by applying for access on the [PetFace page](https://dahlian00.github.io/PetFacePage/).

Extract the `chimp` folder into `./assets/PetFace`:
```bash
tar -xf ./assets/PetFace/chimp.tar.gz -C ./assets/PetFace
```

#### Run the evaluation
```bash
python eval.py \
  --pretrained_weights ./assets/weights/25-08-29T11-49-28_340k.pth \
  --data_root ./assets/PetFace
# Example output:
# ...
# [KNN, k=20]
#   mean_accuracy: 0.4731 ± 0.0183
# ...
# [Summary of 10 runs] Verification ROC-AUC: 0.7680 ± 0.0010
```

The output also reports verification thresholds for the different metrics.

#### Evaluate other baselines
Our evaluation script supports additional baselines that require optional dependencies:
- `MegaDescriptor-L-384` (requires `timm`)
- `miewid-msv2` and `miewid-msv3` (require `transformers`)

Install dependencies (if needed):
```bash
pip install timm transformers
```

Run each baseline separately:
```bash
# MegaDescriptor-L-384
python eval.py --data_root ./assets/PetFace --model_type MegaDescriptor-L-384
# ...
# [KNN, k=7]
#   mean_accuracy: 0.2620 ± 0.0142
# ...
# [Summary of 10 runs] Verification ROC-AUC: 0.7148 ± 0.0008

# miewid-msv3
python eval.py --data_root ./assets/PetFace --model_type miewid-msv3
# ...
# [KNN, k=7]
#   mean_accuracy: 0.4949 ± 0.0115
# ...
# [Summary of 10 runs] Verification ROC-AUC: 0.7453 ± 0.0011

# miewid-msv2
python eval.py --data_root ./assets/PetFace --model_type miewid-msv2
# ...
# [KNN, k=7]
#   mean_accuracy: 0.4641 ± 0.0168
# ...
# [Summary of 10 runs] Verification ROC-AUC: 0.7592 ± 0.0011
```


## License

* The **code** is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* The **pre-trained model weights** are provided under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements

A few shoutouts to the open-source projects and datasets that we used in this work:
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [PetFace dataset](https://dahlian00.github.io/PetFacePage/)
- [PanAf20K dataset](https://data.bris.ac.uk/data/dataset/1h73erszj3ckn2qjwm4sqmr2wt)
- and, of course, PyTorch ecosystem, OpenCV, and other open-source projects that we used in this work (see packages in `requirements.txt`).

This research was funded by EPSRC Programme Grant [VisualAI](https://www.robots.ox.ac.uk/~vgg/projects/visualai/) EP/T028572/1. We thank the Guinean authorities (DGERSIT & IREB), Tetsuro Matsuzawa, Kyoto University, and contributors to the Bossou dataset, [Tacugama Chimpanzee Sanctuary](https://www.tacugama.com/), local authorities, and field staff for access to Loma Mountains National Park camera trap data and support with data sharing and conservation.
