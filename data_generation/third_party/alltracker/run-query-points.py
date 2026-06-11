import torch
import torch.nn.functional as F
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
import os
from prettytable import PrettyTable
import time
import matplotlib.pyplot as plt
import pickle
from matplotlib import colormaps

def read_video(name_path):
    vidcap = cv2.VideoCapture(name_path)
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames

def draw_pts(rgb, pts, visibs, colors, radius=1, inds=None):
    H,W,C = rgb.shape
    assert(C==3)
    N,D = pts.shape
    assert(D==2)
    if inds is not None:
        pts = pts[inds]
        visibs = visibs[inds]
        colors = colors[inds]
    cmap = colormaps['hsv'].resampled(N)
    colors = (cmap(np.arange(N))[:, :3] * 255).astype(np.uint8)  # RGB 0–255
    for ii in range(N):
        xy = pts[ii].round().astype(np.int32)
        # color = (int(colors[ii,0]),int(colors[ii,1]),int(colors[ii,2]))
        color = tuple(int(c) for c in colors[ii][::-1])  # convert RGB to BGR for OpenCV
        if visibs[ii] > 0.5:
            thickness = -1 # filled in
        else:
            thickness = 1 # hollow
        cv2.circle(rgb, (xy[0], xy[1]), radius, color, thickness)
    return rgb

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params

def forward_video(rgbs, model, args, orig_H, orig_W):
    # Use query points to filter out the things we need
    fn = args.file
    cn = args.clip

    # Query points file (.npz) with "query_points" array
    query_path = args.query_path
    data_2dtrack = np.load(query_path, allow_pickle=True)
    
    query_points = data_2dtrack["query_points"]      # (N_queries, 3)
    frame_idx = int(query_points[0, 0])  # get the frame index from the first query point
    query_points = query_points[:, 1:]   # shape (N, 2)
    
    if query_points.size > 0:
        B,T,C,H,W = rgbs.shape
        assert C == 3
        device = rgbs.device
        assert(B==1)

        grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
        grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

        torch.cuda.empty_cache()
        print('starting forward...')
        f_start_time = time.time()

        flows_e, visconf_maps_e, _, _ = \
            model(rgbs[:, frame_idx:], iters=args.inference_iters, sw=None, is_training=False)
        # Edge case: when the forward chunk has only 1 timestep, some codepaths return
        # (B,2,H,W) instead of (B,T,2,H,W). Normalize to 5D and explicitly insert
        # the anchor (zero) flow so the time dimension always includes the query frame.
        if flows_e.dim() == 4:
            flows_e = flows_e.unsqueeze(1)  # B,1,2,H,W (target-only)
            zero_flow = torch.zeros_like(flows_e[:, :1])
            flows_e = torch.cat([zero_flow, flows_e], dim=1)  # B,2,2,H,W (anchor + target)
        if visconf_maps_e.dim() == 4:
            visconf_maps_e = visconf_maps_e.unsqueeze(1)
        if visconf_maps_e.shape[1] == flows_e.shape[1] - 1:
            # Add a full-confidence map for the anchor frame we just inserted.
            anchor_conf = torch.ones_like(visconf_maps_e[:, :1])
            visconf_maps_e = torch.cat([anchor_conf, visconf_maps_e], dim=1)
        traj_maps_e = flows_e + grid_xy # B,Tf,2,H,W
        if frame_idx > 0:
            backward_flows_e, backward_visconf_maps_e, _, _ = \
                model(rgbs[:, :frame_idx+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
            if backward_flows_e.dim() == 4:
                backward_flows_e = backward_flows_e.unsqueeze(1)
                zero_flow = torch.zeros_like(backward_flows_e[:, :1])
                backward_flows_e = torch.cat([zero_flow, backward_flows_e], dim=1)
            if backward_visconf_maps_e.dim() == 4:
                backward_visconf_maps_e = backward_visconf_maps_e.unsqueeze(1)
            if backward_visconf_maps_e.shape[1] == backward_flows_e.shape[1] - 1:
                anchor_conf = torch.ones_like(backward_visconf_maps_e[:, :1])
                backward_visconf_maps_e = torch.cat([anchor_conf, backward_visconf_maps_e], dim=1)
            backward_traj_maps_e = backward_flows_e + grid_xy # B,Tb,2,H,W, reversed
            backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
            backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
            traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1) # B,T,2,H,W
            visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1) # B,T,2,H,W
        ftime = time.time()-f_start_time
        print('finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
        # traj_maps_e = flows_e + grid_xy # B,T,2,H,W
        utils.basic.print_stats('traj_maps_e', traj_maps_e)
        utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

        # if we have a query_points with a dim, use it to scale the query points
        dim_orig = data_2dtrack["dim"]                      # (H_orig, W_orig)
        dim_orig = dim_orig.astype(np.float32)
                       # (H_orig, W_orig)

        # Get shape of traj_maps_e
        B, T, C, H, W = traj_maps_e.shape
        assert B == 1 and C == 2
        print(f"traj_maps_e shape: ({H}, {W})")
        print(f"query poitns in {dim_orig}")

        # Compute scaling factors from original dim to traj_maps_e's spatial size
        scale_y = H / dim_orig[0]
        scale_x = W / dim_orig[1]

        # Scale query points to match H x W of traj_maps_e
        xq = np.round(query_points[:, 0] * scale_x).astype(int)  # width (x)
        yq = np.round(query_points[:, 1] * scale_y).astype(int)  # height (y)
        # yq = np.round(query_points[:, 0]).astype(int)  
        # xq = np.round(query_points[:, 1]).astype(int)  

        # Clamp to valid index ranges
        xq = np.clip(xq, 0, W - 1)
        yq = np.clip(yq, 0, H - 1)

        # Extract tracks and visibility from the full-resolution map
        traj_flat = traj_maps_e[0]         # (T, 2, H, W)
        vis_flat = visconf_maps_e[0]      # (T, 2, H, W)

        # For each query point, index from the flow/trajectory map
        tracks = torch.stack([
            torch.stack([traj_flat[:, 0, y, x], traj_flat[:, 1, y, x]], dim=-1)  # (T, 2)
            for x, y in zip(xq, yq)
        ], dim=1).cpu().numpy()  # Shape: (T, N_queries, 2)

        # Visibility from channel 0
        visibility = torch.stack([
            vis_flat[:, 0, y, x] > 0.5
            for x, y in zip(xq, yq)
        ], dim=1).cpu().numpy()  # Shape: (T, N_queries), bool

        # Rescale tracks back to original video resolution for downstream alignment
        rescale_x = orig_W / W
        rescale_y = orig_H / H
        tracks[:, :, 0] *= rescale_x
        tracks[:, :, 1] *= rescale_y

        out_root = args.out_root
        save_dir = os.path.join(out_root, fn)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{cn}.npz")

        # Save to npz — dim is original resolution to match ViPE depth/intrinsics
        np.savez(save_path, tracks=tracks, visibility=visibility, dim=(orig_H, orig_W))
        print(f"Saved to {save_path}")
        print(tracks.shape)
        print(visibility.shape)
        print((H,W))


        # # Parts below are for visualization:
        # # sort according to velocity, so that moving points are drawn last
        # # vels = tracks[1:] - tracks[:-1] # T-1,N,2
        # # vels = np.linalg.norm(vels, axis=-1).mean(axis=0)
        # # inds = np.argsort(vels)
        # colors = None

        # # fn = args.mp4_path.split('/')[-1].split('.')[0]
        # # Stereo4d
        # rgb_out_f = f'/path/to/data/HD-EPIC/ssv2-videos/{cn}/pt_vis_%s_q%d.mp4' % (fn, frame_idx)
        # print('rgb_out_f', rgb_out_f)
        # temp_dir = f'/path/to/data/HD-EPIC/ssv2-videos/{cn}/temp_pt_vis_%s_q%d' % (fn, frame_idx)
        # utils.basic.mkdir(temp_dir)
        # vis = []
        # for ti in range(T):
        #     pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
        #                     tracks[ti],
        #                     visibility[ti],
        #                     colors=colors,
        #                     radius=4)
        #     vis.append(pt_vis)
        # for ti in range(T):
        #     temp_out_f = '%s/%03d.png' % (temp_dir, ti)
        #     im = PIL.Image.fromarray(vis[ti])
        #     im.save(temp_out_f, "PNG", subsampling=0, quality=100)
        # # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 30 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))

        # import subprocess
        # import imageio_ffmpeg

        # # Get the path to the bundled ffmpeg executable
        # ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        # # Build the command
        # cmd = [
        #     ffmpeg_path, '-y', '-hide_banner', '-loglevel', 'error',
        #     '-f', 'image2', '-framerate', '30',
        #     '-pattern_type', 'glob', '-i', f'{temp_dir}/*.png',
        #     # libx264+yuv420p require even width/height; pad by 1px if needed.
        #     '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        #     '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
        #     rgb_out_f
        # ]

        # # Run the command
        # subprocess.run(cmd, check=True)
        
        return None
    else:
        print("query_points is empty, skipping.")
        return None

def run(model, args):
    log_dir = './logs_demo'
    
    global_step = 0

    if args.ckpt_init:
        _ = utils.saveload.load(
            None,
            args.ckpt_init,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        print('loaded weights from', args.ckpt_init)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        print('loaded weights from', url)

    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    
    # Video path
    video_path = args.video_path

    rgbs = read_video(video_path)

    H,W = rgbs[0].shape[:2]
    
    # shorten & shrink the video, in case the gpu is small
    max_frames = int(getattr(args, "max_frames", 400))
    if max_frames > 0:
        rgbs = rgbs[:max_frames]

    orig_H, orig_W = H, W
    HH = int(getattr(args, "max_side", 1024))
    scale = min(HH/H, HH/W)
    if scale < 1.0:
        H, W = int(H*scale), int(W*scale)
        H, W = H//8 * 8, W//8 * 8 # make it divisible by 8
        rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        print(f'Resized from ({orig_H},{orig_W}) to ({H},{W}) for inference (max_side={HH})')
    else:
        print(f'No resize needed: ({H},{W}) already within max_side={HH}')

    # move to gpu
    rgbs = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float() # 1,T,C,H,W
    print('rgbs', rgbs.shape)
    
    with torch.no_grad():
        metrics = forward_video(rgbs, model, args, orig_H, orig_W)
    
    return None

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_init", type=str, default='') # the ckpt we want (else default)
    parser.add_argument("--file", type=str, default='bear') # input video
    parser.add_argument("--clip", type=str, default='bear') # input video
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="Absolute path to the input video (overrides dataset-specific defaults).",
    )
    parser.add_argument(
        "--query_path",
        type=str,
        default="",
        help="Absolute path to the query-points .npz (overrides dataset-specific defaults).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./2d_tracks/",
        help="Root output directory for saved 2D tracks (saved as <out_root>/<file>/<clip>.npz).",
    )
    parser.add_argument("--max_frames", type=int, default=400, help="If >0, only process the first N frames.")
    parser.add_argument("--max_side", type=int, default=1024, help="Resize so max(H,W) == max_side before inference.")
    parser.add_argument("--inference_iters", type=int, default=4) # number of inference steps per forward
    parser.add_argument("--window_len", type=int, default=16) # model hyperparam
    parser.add_argument("--subsample_rate", type=int, default=16) # vis hyp
    parser.add_argument("--mixed_precision", action='store_true', default=False)
    parser.add_argument("--mask", action='store_true', default=False)
    args = parser.parse_args()

    # Normalize optional args
    if args.video_path == "":
        args.video_path = None
    if args.query_path == "":
        args.query_path = None

    from nets.alltracker import Net; model = Net(args.window_len)
    count_parameters(model)

    run(model, args)
    
