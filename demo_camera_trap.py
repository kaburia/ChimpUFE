from pathlib import Path
import subprocess
import argparse
import os
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
import torchvision
import torch

from src.tracker.yolox.utils import fuse_model
from src.tracker.yolox.tracker.byte_tracker import BYTETracker
from src.tracker.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


def get_args():
    parser = argparse.ArgumentParser("Camera Trap Demo")
    parser.add_argument("--video_path", default="./assets/IMG_0240.MP4", help="path to images or video")
    parser.add_argument("--overlay_results_on_video", action="store_true", help="overlay tracking results on video")
    parser.add_argument("--save_results_path", type=str, default=None, help="the tracking result video save path")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--pretrained_weights", default=None, type=str, help="pretrained_weights for eval")

    # tracking args
    parser.add_argument( "--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument( "--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
    help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    args = parser.parse_args()
    return args


def get_model(ckpt_file, fuse, fp16, depth, width, in_channels, num_classes, device):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    pretrained_weights = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(pretrained_weights["model"])
    model = model.to(device)
    model.eval()
    if fuse:
        print("\tFusing model...")
        model = fuse_model(model)
    return model


class YoloXPredictor(object):
    def __init__(
        self,
        model,
        num_classes,
        confthre,
        nmsthre,
        test_size,
        cls_names=('chimp', ),
        device=torch.device("cpu"),
        fp16=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = num_classes
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.test_size = test_size
        self.device = device
        self.fp16 = fp16

    def inference(self, img, verbose=True):
        height, width = img.shape[:2]
        img_info = {
            "id": 0,
            "file_name": None,
            "height": height,
            "width": width,
            "raw_img": img,
        }
        img, _ = self.preprocess(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().to(self.device)

        with torch.set_grad_enabled(False), torch.autocast('cuda', enabled=self.device.type == 'cuda' and self.fp16):
            t0 = time.time()
            outputs = self.model(img)
            outputs = self.postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            if verbose:
                print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def preprocess(self, img, res, input_size):
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        new_shape = int(img.shape[1] * r), int(img.shape[0] * r)
        resized_img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[:new_shape[1], :new_shape[0]] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, np.zeros((1, 5))

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            dets = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            dets = dets[conf_mask]
            if not dets.size(0):
                continue

            nms_out_index = torchvision.ops.nms(dets[:, :4], dets[:, 4] * dets[:, 5], nms_thre,)

            dets = dets[nms_out_index]
            if output[i] is None:
                output[i] = dets
            else:
                output[i] = torch.cat((output[i], dets))

        return output


def process_video(predictor: YoloXPredictor, tracker: BYTETracker, args, test_size):
    '''Process the video frame-by-frame, running prediction and tracking,
    and record the raw results in a dictionary keyed by frame id.'''
    cap = cv2.VideoCapture(args.video_path)
    frame_id = 0
    results_dict = {}  # frame_id -> list of detections
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        # Inference & tracking
        outputs, img_info = predictor.inference(frame, verbose=False)
        frame_detections = []
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    frame_detections.append({'tlwh': tlwh, 'track_id': tid, 'score': t.score})
        # Save the detections for this frame
        results_dict[frame_id] = {'img_info': img_info, 'detections': frame_detections }
        frame_id += 1
        pbar.update(1)
    pbar.close()

    # factorize the track ids from the the results
    old2new = {}
    for frame_id, info in results_dict.items():
        for det in info['detections']:
            tid = det['track_id']
            if tid not in old2new:
                old2new[tid] = len(old2new) + 1
            det['track_id'] = old2new[tid]
    return results_dict



def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, ids2=None):
    def get_color(idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color
    im = np.ascontiguousarray(np.copy(image))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def visualize_results(input_video_path, results_dict, save_vid_path: Path):
    '''Overlay tracking results on each frame using the recorded results,
    and save the visualized frames to an MP4 file.'''
    cap = cv2.VideoCapture(input_video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create directory if it doesn't exist
    save_vid_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Making a visualization video...")

    vid_writer = cv2.VideoWriter(str(save_vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    frame_id = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Overlaying tracking results")
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        # Get the stored info for this frame. Fallback to original frame if not available.
        if frame_id in results_dict:
            # img_info = results_dict[frame_id]['img_info']
            detections = results_dict[frame_id]['detections']
            # Collect detections for plotting.
            online_tlwhs = [det['tlwh'] for det in detections]
            online_ids = [det['track_id'] for det in detections]
            # Overlay the detections.
            # Note: You can adjust the plotting function as needed.
            online_im = frame
            online_im = plot_tracking(online_im, online_tlwhs, online_ids, frame_id=frame_id + 1)
        else:
            online_im = frame
        vid_writer.write(online_im)
        frame_id += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    vid_writer.release()

    # Compress the video and save it.
    save_vid_path = Path(save_vid_path).resolve()
    # rename to uncompr_ prefix
    compr_vid_path = save_vid_path.parent / f'{save_vid_path.stem}.mp4'
    save_vid_path = save_vid_path.rename(save_vid_path.parent / f'uncompr_{save_vid_path.name}')
    print(f"Compressing video...")
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-stats',
                    '-i', str(save_vid_path), '-vcodec', 'libx264', '-crf', '28', '-y', str(compr_vid_path)])
    print(f"Compressed video saved to {compr_vid_path}")
    # remove the original video
    os.remove(save_vid_path)


def main(args):
    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    num_classes = 1
    in_channels = [256, 512, 1024]
    depth = 1.33
    width = 1.25
    test_conf = 0.001
    nmsthre = 0.7
    test_size = (800, 1440)

    model = get_model(args.pretrained_weights, args.fuse, args.fp16, depth, width, in_channels, num_classes, device)

    # paths
    video_path = Path(args.video_path)
    save_results_path = Path(args.save_results_path)
    vis_folder = save_results_path / 'vis'
    vis_folder.parent.mkdir(parents=True, exist_ok=True)
    save_vid_path = Path(vis_folder) / video_path.name
    save_txt_path = save_results_path / video_path.with_suffix('.txt').name
    save_txt_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(save_txt_path).exists():
        print(f"File {save_txt_path} already exists. Skipping...")
        return

    predictor = YoloXPredictor(model, num_classes, test_conf, nmsthre, test_size, ('chimp', ), device, args.fp16)
    tracker = BYTETracker(args, frame_rate=30)
    results = process_video(predictor, tracker, args, test_size)

    # Save the results to a text file.
    with open(save_txt_path, 'w') as f:
        for frame_id, info in results.items():
            for det in info['detections']:
                tlwh = det['tlwh']
                tid = det['track_id']
                score = det['score']
                f.write(f'{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n')
    print(f"Tracking results saved to {save_txt_path}")
    print('Columns: frame_id, track_id, top, left, width, height, score, -1, -1, -1')

    # Visualize the results
    if args.overlay_results_on_video:
        visualize_results(args.video_path, results, save_vid_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
