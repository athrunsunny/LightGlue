from pathlib import Path
import argparse
from models_lightglue import LightGlue, SuperPoint, DISK, ALIKED
from models_lightglue.utils import load_image, rbd

import torch
import cv2
import matplotlib.cm as cm
import numpy as np


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    # out = 255*np.ones((H, W), np.uint8)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0 + margin:, :] = image1
    # out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def parser_opt(input, input_im1, input_im2, resize, match_threshold, keypoint_threshold):
    parser = argparse.ArgumentParser(description='SuperLightGlue demo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default=input,
                        help='ID of a USB webcam, URL of an IP camera, or path to an image directory or movie file')
    parser.add_argument('--input_im1', type=str, default=input_im1, help='path1 to an image directory')
    parser.add_argument('--input_im2', type=str, default=input_im2, help='path2 to an image directory')
    parser.add_argument('--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
                        help='Glob if a directory of images is specified')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is a movie or directory')
    parser.add_argument('--max_length', type=int, default=1000000,
                        help='Maximum length if input is a movie or directory')
    parser.add_argument('--resize', type=int, nargs='+', default=resize,
                        help='Resize the input image before running inference. If two numbers, '
                             'resize to the exact dimensions, if one number, resize the max '
                             'dimension, if -1, do not resize')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor',
                        help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=-1,
                        help='Maximum number of keypoints detected by Superpoint(\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=keypoint_threshold,
                        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4,
                        help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=match_threshold,
                        help='SuperGlue match threshold default 0.2')
    parser.add_argument('--show_keypoints', action='store_true', default=True,
                        help='Show the detected keypoints')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force pytorch to run in CPU mode.')
    # parser.add_argument(
    #     '--output_path', type=str, default=r'',
    #     help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    # 对文件中的两个指定图像做关键点匹配，并将匹配好的关键点保存到对应的npy文件中，注意resize的尺寸，以及match_threshold
    input = r'G:\point_match\calibrate\camera_test_gt_val\20240529tc\3604\5555'
    input_im1 = fr'{input}\u30.jpg'
    input_im2 = fr'{input}\un.jpg'
    resize = [640, 480]
    # resize = [1920, 1080]
    match_threshold = 0.01
    keypoint_threshold = 0.0005

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    opt = parser_opt(input, input_im1, input_im2, resize, match_threshold, keypoint_threshold)
    save_dir = opt.input

    extractor = SuperPoint(max_num_keypoints=None, detection_threshold=opt.keypoint_threshold).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    image0 = load_image(opt.input_im1, resize=[opt.resize[1], opt.resize[0]])
    image1 = load_image(opt.input_im2, resize=[opt.resize[1], opt.resize[0]])

    image0_gray = cv2.imread(opt.input_im1)
    image1_gray = cv2.imread(opt.input_im2)
    image0_gray = cv2.resize(image0_gray, opt.resize)
    image1_gray = cv2.resize(image1_gray, opt.resize)

    feats0 = extractor.extract(image0.to(device), resize=[opt.resize[1], opt.resize[0]])
    feats1 = extractor.extract(image1.to(device), resize=[opt.resize[1], opt.resize[0]])
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"].cpu().numpy(), feats1["keypoints"].cpu().numpy(), matches01[
        "matches"].cpu().numpy()
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    np.save(f'{save_dir}/kp1_splg.npy', m_kpts0)
    np.save(f'{save_dir}/kp2_splg.npy', m_kpts1)

    confidence = matches01["scores"].cpu().numpy()
    color = cm.jet(confidence)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(m_kpts0))
    ]
    k_thresh = opt.keypoint_threshold
    m_thresh = opt.match_threshold
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]
    out = make_matching_plot_fast(
        image0_gray, image1_gray, kpts0, kpts1, m_kpts0, m_kpts1, color, text,
        path=None, show_keypoints=True, small_text=small_text)

    cv2.imwrite(rf"{save_dir}/resized_match_points_splg.jpg", out)

    print('match done!')
