# Universal Chimp Face Embedder

```bibtex
@InProceedings{self2025iashin,
  title={Self-supervised Learning on Camera Trap Footage Yields a Strong Universal Face Embedder},
  author={Iashin, V., Lee, H., Schofield, D., and Zisserman, A.},
  booktitle={ICIP Workshop on Computer Vision for Ecological and Biodiversity Monitoring},
  year={2025},
  organization={IEEE}
}
```

[Project Page](https://www.robots.ox.ac.uk/~vgg/research/ChimpUFE/) • [Paper (to be added)]

This repo provides two main functionalities at the moment:
1. **Face Recognition**: Compare a query image to a gallery of images using a Universal Face Embedder model to find similar faces or compute similarity scores.
2. **Camera Trap Detection**: Detect and track animals (chimps) in video footage using YOLOX object detection with ByteTracker for multi-object tracking.

## Installation
Install the required packages in your python environment using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Files and Folders
- `demo_face_rec.py`: Main script for face recognition inference.
- `demo_camera_trap.py`: Main script for camera trap detection and tracking.
- `assets/`: Example images, videos, gallery structure, and pre-trained model weights.
- `src/face_embedder/`: Source code for the Universal Chimp Face Embedder.
- `src/tracker/`: Source code for YOLOX object detection and ByteTracker tracking.

## Pre-trained Weights

Before running the demos, download the pre-trained weights for the ViT-Base version from [here (415 MB)](https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/25-06-06T20-51-36_330k.pth) and place them in the `./assets/weights` folder:
```bash
# face recognition weights
wget -P ./assets/weights https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/25-06-06T20-51-36_330k.pth
```
and if you'd like to run the camera trap demo, download the YOLOX weights from [here (378 MB)](https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/yolox_best_only_model.pth)
```bash
# camera trap weights
wget -P ./assets/weights https://github.com/v-iashin/ChimpUFE/releases/download/v1.0/yolox_best_only_model.pth
```

**Note**: All pre-trained weights are licensed under non-commercial use only.

## Usage

### 1. Compare Two Images Directly
This prints the similarity score between two images:
```bash
python demo_face_rec.py \
  --pretrained_weights ./assets/weights/25-06-06T20-51-36_330k.pth \
  --query_path ./assets/gallery/000000/01.png \
  --gallery_path ./assets/gallery/000001/00.png

# Cosine similarity between the query and the gallery image: 0.1976
```

### 2. Compare a Query Image to a Gallery Folder
This finds the top-k most similar images in the gallery to your query image.
The gallery folder should be structured as subfolders (e.g., `gallery/000000/`, `gallery/000001/`, etc.) each containing images.

```bash
python demo_face_rec.py \
  --pretrained_weights ./assets/weights/25-06-06T20-51-36_330k.pth \
  --query_path ./assets/gallery/000003/00.png \
  --gallery_path ./assets/gallery \
  --top_k 10

# Top 10 matches for query './assets/gallery/000003/00.png':
#  1. ./assets/gallery/000003/00.png | similarity: 1.0000
#  2. ./assets/gallery/000003/01.png | similarity: 0.6378
#  3. ./assets/gallery/000003/03.png | similarity: 0.5946
#  4. ./assets/gallery/000003/04.png | similarity: 0.5346
#  5. ./assets/gallery/000003/02.png | similarity: 0.5069
#  6. ./assets/gallery/000002/02.png | similarity: 0.4838
#  7. ./assets/gallery/000002/00.png | similarity: 0.4723
#  8. ./assets/gallery/000002/03.png | similarity: 0.4323
#  9. ./assets/gallery/000002/01.png | similarity: 0.3673
# 10. ./assets/gallery/000002/04.png | similarity: 0.3620
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
# Columns: frame_id, track_id, top, left, width, height, score, -1, -1, -1
# 0,1,594.00,255.60,110.70,117.45,0.95,-1,-1,-1
# 0,2,179.77,248.17,26.10,49.50,0.87,-1,-1,-1
# 1,1,604.48,254.04,109.26,115.89,0.95,-1,-1,-1
# 1,2,179.77,248.17,26.10,49.50,0.87,-1,-1,-1
# 2,1,616.73,252.01,109.26,115.62,0.95,-1,-1,-1
# 2,2,179.95,248.17,25.92,49.15,0.87,-1,-1,-1
# 3,1,632.11,249.68,110.54,116.94,0.95,-1,-1,-1
# 3,2,180.00,248.17,25.87,49.04,0.87,-1,-1,-1
# 4,1,648.36,247.35,113.33,120.03,0.95,-1,-1,-1
# 4,2,180.01,248.17,25.87,49.01,0.87,-1,-1,-1
# 5,1,666.79,245.99,116.75,124.23,0.95,-1,-1,-1
```


The results will be saved in the `assets/detections` folder.

If you want to overlay the tracking results on the video, you can add the `--overlay_results_on_video` flag
and find the results in `./assets/detections/vis`.


## License

* The **code** is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* The **pre-trained model weights** are provided under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements

This research was funded by EPSRC Programme Grant VisualAI EP/T028572/1. We thank the Guinean authorities (DGERSIT & IREB), T. Matsuzawa, Kyoto University, and contributors to the Bossou dataset, Tacugama Chimpanzee Sanctuary, local authorities, and field staff for access to Loma Mountains camera trap data and support with data sharing and conservation.
