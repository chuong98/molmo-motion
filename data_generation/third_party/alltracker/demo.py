# import torch
# import torch.nn.functional as F
# import cv2
# import argparse
# import utils.saveload
# import utils.basic
# import utils.improc
# import PIL.Image
# import numpy as np
# import os
# from prettytable import PrettyTable
# import time
# import matplotlib.pyplot as plt

# def read_mp4(name_path):
#     vidcap = cv2.VideoCapture(name_path)
#     frames = []
#     while vidcap.isOpened():
#         ret, frame = vidcap.read()
#         if ret == False:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#     vidcap.release()
#     return frames

# def draw_pts(rgb, pts, visibs, confs, colors, radius=2, conf_thr=0.1, inds=None):
#     H,W,C = rgb.shape
#     assert(C==3)
#     N,D = pts.shape
#     assert(D==2)
#     if inds is not None:
#         pts = pts[inds]
#         visibs = visibs[inds]
#         confs = confs[inds]
#         # colors = colors[inds]
#     fixed_color = (255, 0, 0)  # red in BGR
#     for ii in range(N):
#         xy = pts[ii].round().astype(np.int32)
#         # color = (int(colors[ii,0]),int(colors[ii,1]),int(colors[ii,2]))
#         color = fixed_color
#         if visibs[ii] > 0.5:
#             thickness = -1 # filled in
#         else:
#             thickness = 1 # hollow
#         if confs[ii] > conf_thr:
#             cv2.circle(rgb, (xy[0], xy[1]), radius, color, thickness)
#     return rgb

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         param = parameter.numel()
#         if param > 100000:
#             table.add_row([name, param])
#         total_params+=param
#     print(table)
#     print('total params: %.2f M' % (total_params/1000000.0))
#     return total_params

# def forward_video(rgbs, model, args, orig_H, orig_W):
#     B,T,C,H,W = rgbs.shape
#     fn = args.file
#     assert C == 3
#     device = rgbs.device
#     assert(B==1)

#     grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
#     grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

#     torch.cuda.empty_cache()
#     print('starting forward...')
#     f_start_time = time.time()

#     flows_e, visconf_maps_e, _, _ = \
#         model(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None, is_training=False)
#     traj_maps_e = flows_e + grid_xy # B,Tf,2,H,W
#     if args.query_frame > 0:
#         backward_flows_e, backward_visconf_maps_e, _, _ = \
#             model(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
#         backward_traj_maps_e = backward_flows_e + grid_xy # B,Tb,2,H,W, reversed
#         backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
#         backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
#         traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1) # B,T,2,H,W
#         visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1) # B,T,2,H,W
#     ftime = time.time()-f_start_time
#     print('finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
#     # traj_maps_e = flows_e + grid_xy # B,T,2,H,W
#     utils.basic.print_stats('traj_maps_e', traj_maps_e)
#     utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

#     # subsample to make the vis more readable
#     rate = args.subsample_rate
#     trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
#     visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2

#     # segment out the area
#     if args.mask == True:
#         mask_path=f"/path/to/data/DAVIS/Annotations_unsupervised/480p/{fn}/00000.png"
#         print(f'Loading mask from {mask_path}')
#         mask_img = cv2.imread(mask_path)
#         mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
#         mask_resized = cv2.resize(mask_rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
#         mask_binary = np.any(mask_resized != [0, 0, 0], axis=-1)
#         mask_subsampled = mask_binary[::rate, ::rate].flatten()
#         trajs_e = trajs_e[:, :, mask_subsampled]
#         visconfs_e = visconfs_e[:, :, mask_subsampled]

#     print('trajs_e', trajs_e.shape)
#     # add a saving track feature
#     # Get shape
#     B, T, N, _ = trajs_e.shape
#     assert B == 1

#     # Convert to numpy
#     tracks = trajs_e[0].cpu().numpy()  # (T, N, 2)
#     visibility = visconfs_e[0,:,:,0].cpu().numpy() > 0.5  # (T, N) -> boolean mask

#     # Compute actual scale factors used for height and width
#     scale_h = H / orig_H  # scaled height / original height
#     scale_w = W / orig_W  # scaled width / original width

#     # Apply inverse scaling to x (width) and y (height) separately
#     tracks_rescaled = tracks.copy()
#     tracks_rescaled[..., 0] = tracks[..., 0] / scale_w  # x-coords
#     tracks_rescaled[..., 1] = tracks[..., 1] / scale_h  # y-coords
#     dim_rescaled = (orig_H, orig_W)

#     # Stereo4d
#     # save_dir = os.path.join("2d-track-filtered/stereo4d", fn)
#     # davis
#     save_dir = os.path.join("2d-track/davis", fn)
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(save_dir, f"{fn}.npz")

#     # Save to npz
#     np.savez(save_path, tracks=tracks_rescaled, visibility=visibility, dim=dim_rescaled)
#     print(f"Saved to {save_path}")
#     print(tracks_rescaled.shape)
#     print(visibility.shape)
#     print(dim_rescaled)

#     xy0 = trajs_e[0,0].cpu().numpy()
#     colors = utils.improc.get_2d_colors(xy0, H, W)

#     # sort according to velocity, so that moving points are drawn last
#     vels = trajs_e[0,1:].detach().cpu().numpy() - trajs_e[0,:-1].detach().cpu().numpy() # T-1,N,2
#     vels = np.linalg.norm(vels, axis=-1).mean(axis=0)
#     inds = np.argsort(vels)

#     # fn = args.mp4_path.split('/')[-1].split('.')[0]
#     # Stereo4d
#     rgb_out_f = f'./output-videos/davis/{fn}/pt_vis_%s_rate%d_q%d.mp4' % (fn, rate, args.query_frame)
#     print('rgb_out_f', rgb_out_f)
#     temp_dir = f'./output-videos/davis/{fn}/temp_pt_vis_%s_rate%d_q%d' % (fn, rate, args.query_frame)
#     utils.basic.mkdir(temp_dir)
#     vis = []
#     for ti in range(T):
#         # pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
#         #                   trajs_e[0,ti].detach().cpu().numpy(),
#         #                   visconfs_e[0,ti,:,0].detach().cpu().numpy(),
#         #                   visconfs_e[0,ti,:,1].detach().cpu().numpy(),
#         #                   colors=colors,
#         #                   radius=max(int(rate//2),1),
#         #                   inds=inds)
#         pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
#                           trajs_e[0,ti].detach().cpu().numpy(),
#                           visconfs_e[0,ti,:,0].detach().cpu().numpy(),
#                           visconfs_e[0,ti,:,1].detach().cpu().numpy(),
#                           colors=colors,
#                           radius=4,
#                           inds=inds)
#         vis.append(pt_vis)
#     for ti in range(T):
#         temp_out_f = '%s/%03d.png' % (temp_dir, ti)
#         im = PIL.Image.fromarray(vis[ti])
#         im.save(temp_out_f, "PNG", subsampling=0, quality=100)
#     # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 30 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))

#     import subprocess
#     import imageio_ffmpeg

#     # Get the path to the bundled ffmpeg executable
#     ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

#     # Build the command
#     cmd = [
#         ffmpeg_path, '-y', '-hide_banner', '-loglevel', 'error',
#         '-f', 'image2', '-framerate', '30',
#         '-pattern_type', 'glob', '-i', f'{temp_dir}/*.png',
#         '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
#         rgb_out_f
#     ]

#     # Run the command
#     subprocess.run(cmd, check=True)

#     # Run cycle-consistent tracking filtering
#     print("Running cycle-consistent backward tracking...")
#     tracked_pts_last = trajs_e[0, -1].detach().cpu()  # [N, 2]
#     rgbs_rev = rgbs.flip(1)  # filp the video
#     with torch.no_grad():
#         flows_b, _, _, _ = model(rgbs_rev, iters=args.inference_iters, sw=None, is_training=False)
#     traj_maps_b = flows_b + grid_xy  # [1, T, 2, H, W]
#     traj_maps_b = traj_maps_b.flip(1)

#     grid_flat = grid_xy[0, 0].permute(1, 2, 0).reshape(-1, 2).cpu() # [H*W, 2]
#     dists = torch.cdist(tracked_pts_last.unsqueeze(0), grid_flat.unsqueeze(0)).squeeze(0)  # [N, H*W]
#     indices = dists.argmin(dim=1)  # [N]

#     traj_maps_b = traj_maps_b.view(B, T, 2, -1)  # [1, T, 2, H*W]
#     trajs_b = traj_maps_b[0, :, :, indices]  # [T, 2, N]
#     trajs_b = trajs_b.permute(0, 2, 1)  # [T, N, 2]

#     start_fwd = trajs_e[0, 0].detach().cpu()  # [N, 2]
#     # print("fwd: ", start_fwd)
#     end_back = trajs_b[0].detach().cpu()  # [N, 2]
#     # print("end: ", end_back)
#     cycle_dists = torch.norm(start_fwd - end_back, dim=1)  # [N]
#     # print(cycle_dists)

#     cycle_thresh = 5
#     valid_mask = cycle_dists < cycle_thresh
#     print(f"Cycle-consistency: kept {valid_mask.sum().item()} / {len(valid_mask)} points")

#     # # # DEBUG!!!
#     # Convert image to numpy RGB [H, W, 3]
#     img_np = rgbs[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

#     # Round and convert coordinates to int
#     start_pts = start_fwd.numpy().round().astype(np.int32)   # green
#     end_pts = end_back.numpy().round().astype(np.int32)      # red

#     # Draw start_fwd (green)
#     for pt in start_pts:
#         cv2.circle(img_np, tuple(pt), radius=3, color=(0, 255, 0), thickness=-1)

#     # Draw end_back (red)
#     for pt in end_pts:
#         cv2.circle(img_np, tuple(pt), radius=3, color=(255, 0, 0), thickness=1)

#     selected_pts = start_pts[valid_mask].astype(int)
#     for pt in selected_pts:
#         cv2.circle(img_np, tuple(pt), radius=3, color=(0, 0, 255), thickness=1)

#     # Save the output image
#     cv2.imwrite(f"./output/compare-{fn}.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
#     print(f"Saved comparison image to ./output/compare-{fn}.png")
#     # # # DEBUG!!!


#     log_path = "alltracker_cycle_consistency_log.txt"
#     with open(log_path, "a") as f:
#         f.write(f"{args.file}: kept {valid_mask.sum().item()} / {len(valid_mask)} points\n")
#     trajs_e = trajs_e[:, :, valid_mask]
#     visconfs_e = visconfs_e[:, :, valid_mask]

#     # Convert to numpy
#     tracks = trajs_e[0].cpu().numpy()  # (T, N, 2)
#     visibility = visconfs_e[0,:,:,0].cpu().numpy() > 0.5  # (T, N) -> boolean mask

#     # Compute actual scale factors used for height and width
#     scale_h = H / orig_H  # scaled height / original height
#     scale_w = W / orig_W  # scaled width / original width

#     # Apply inverse scaling to x (width) and y (height) separately
#     tracks_rescaled = tracks.copy()
#     tracks_rescaled[..., 0] = tracks[..., 0] / scale_w  # x-coords
#     tracks_rescaled[..., 1] = tracks[..., 1] / scale_h  # y-coords
#     dim_rescaled = (orig_H, orig_W)

#     # Stereo4d
#     # save_dir = os.path.join("2d-track-filtered/stereo4d", fn)
#     # davis
#     save_dir = os.path.join("2d-track-filtered/davis", fn)
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(save_dir, f"{fn}.npz")

#     # Save to npz
#     np.savez(save_path, tracks=tracks_rescaled, visibility=visibility, dim=dim_rescaled)
#     print(f"Saved to {save_path}")
#     print(tracks_rescaled.shape)
#     print(visibility.shape)
#     print(dim_rescaled)

#     xy0 = trajs_e[0,0].cpu().numpy()
#     colors = utils.improc.get_2d_colors(xy0, H, W)

#     # sort according to velocity, so that moving points are drawn last
#     vels = trajs_e[0,1:].detach().cpu().numpy() - trajs_e[0,:-1].detach().cpu().numpy() # T-1,N,2
#     vels = np.linalg.norm(vels, axis=-1).mean(axis=0)
#     inds = np.argsort(vels)

#     # fn = args.mp4_path.split('/')[-1].split('.')[0]
#     # Stereo4d
#     rgb_out_f = f'./output-videos-filtered/davis/{fn}/pt_vis_%s_rate%d_q%d.mp4' % (fn, rate, args.query_frame)
#     print('rgb_out_f', rgb_out_f)
#     temp_dir = f'./output-videos-filtered/davis/{fn}/temp_pt_vis_%s_rate%d_q%d' % (fn, rate, args.query_frame)
#     utils.basic.mkdir(temp_dir)
#     vis = []
#     for ti in range(T):
#         # pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
#         #                   trajs_e[0,ti].detach().cpu().numpy(),
#         #                   visconfs_e[0,ti,:,0].detach().cpu().numpy(),
#         #                   visconfs_e[0,ti,:,1].detach().cpu().numpy(),
#         #                   colors=colors,
#         #                   radius=max(int(rate//2),1),
#         #                   inds=inds)
#         pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
#                           trajs_e[0,ti].detach().cpu().numpy(),
#                           visconfs_e[0,ti,:,0].detach().cpu().numpy(),
#                           visconfs_e[0,ti,:,1].detach().cpu().numpy(),
#                           colors=colors,
#                           radius=4,
#                           inds=inds)
#         vis.append(pt_vis)
#     for ti in range(T):
#         temp_out_f = '%s/%03d.png' % (temp_dir, ti)
#         im = PIL.Image.fromarray(vis[ti])
#         im.save(temp_out_f, "PNG", subsampling=0, quality=100)
#     # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 30 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))

#     import subprocess
#     import imageio_ffmpeg

#     # Get the path to the bundled ffmpeg executable
#     ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

#     # Build the command
#     cmd = [
#         ffmpeg_path, '-y', '-hide_banner', '-loglevel', 'error',
#         '-f', 'image2', '-framerate', '30',
#         '-pattern_type', 'glob', '-i', f'{temp_dir}/*.png',
#         '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
#         rgb_out_f
#     ]

#     # Run the command
#     subprocess.run(cmd, check=True)
#     # # flow vis
#     # rgb_out_f = './flow_vis.mp4'
#     # temp_dir = 'temp_flow_vis'
#     # utils.basic.mkdir(temp_dir)
#     # vis = []
#     # for ti in range(T):
#     #     flow_vis = utils.improc.flow2color(flows_e[0:1,ti])
#     #     vis.append(flow_vis)
#     # for ti in range(T):
#     #     temp_out_f = '%s/%03d.png' % (temp_dir, ti)
#     #     im = PIL.Image.fromarray(vis[ti][0].permute(1,2,0).cpu().numpy())
#     #     im.save(temp_out_f, "PNG", subsampling=0, quality=100)
#     # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 24 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 1 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))
    
#     return None

# def run(model, args):
#     log_dir = './logs_demo'
    
#     global_step = 0

#     if args.ckpt_init:
#         _ = utils.saveload.load(
#             None,
#             args.ckpt_init,
#             model,
#             optimizer=None,
#             scheduler=None,
#             ignore_load=None,
#             strict=True,
#             verbose=False,
#             weights_only=False,
#         )
#         print('loaded weights from', args.ckpt_init)
#     else:
#         url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
#         model.load_state_dict(state_dict['model'], strict=True)
#         print('loaded weights from', url)

#     model.cuda()
#     for n, p in model.named_parameters():
#         p.requires_grad = False
#     model.eval()
#     # davis
#     mp4_path = f"/path/to/data/DAVIS/JPEGImages/480p/videos/{args.file}.mp4"

#     # stereo4d
#     # mp4_path = f"/path/to/data/dataset/stereo4d/{args.file}/{args.file}.mp4"
#     rgbs = read_mp4(mp4_path)
#     print('rgbs[0]', rgbs[0].shape)
#     H,W = rgbs[0].shape[:2]
#     orig_H,orig_W = rgbs[0].shape[:2]
    
#     # shorten & shrink the video, in case the gpu is small
#     rgbs = rgbs[:400]
#     # Here!
#     HH = 1024
#     scale = min(HH/H, HH/W)
#     H, W = int(H*scale), int(W*scale)
#     H, W = H//8 * 8, W//8 * 8 # make it divisible by 8
#     rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
#     print('rgbs[0]', rgbs[0].shape)

#     # move to gpu
#     rgbs = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
#     rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float() # 1,T,C,H,W
#     print('rgbs', rgbs.shape)
    
#     with torch.no_grad():
#         metrics = forward_video(rgbs, model, args, orig_H, orig_W)
    
#     return None

# if __name__ == "__main__":
#     torch.set_grad_enabled(False)
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt_init", type=str, default='') # the ckpt we want (else default)
#     # parser.add_argument("--mp4_path", type=str, default='./demo_video/monkey.mp4') # input video
#     parser.add_argument("--file", type=str, default='bear') # input video
#     parser.add_argument("--query_frame", type=int, default=0) # which frame to track from
#     parser.add_argument("--inference_iters", type=int, default=4) # number of inference steps per forward
#     parser.add_argument("--window_len", type=int, default=16) # model hyperparam
#     parser.add_argument("--subsample_rate", type=int, default=16) # vis hyp
#     parser.add_argument("--mixed_precision", action='store_true', default=False)
#     parser.add_argument("--mask", action='store_true', default=False)
#     args = parser.parse_args()

#     from nets.alltracker import Net; model = Net(args.window_len)
#     count_parameters(model)

#     run(model, args)
    
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

def read_mp4(name_path):
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

def draw_pts(rgb, pts, visibs, confs, colors, radius=2, conf_thr=0.1, inds=None):
    H,W,C = rgb.shape
    assert(C==3)
    N,D = pts.shape
    assert(D==2)
    if inds is not None:
        pts = pts[inds]
        visibs = visibs[inds]
        confs = confs[inds]
        # colors = colors[inds]
    fixed_color = (255, 0, 0)  # red in BGR
    for ii in range(N):
        xy = pts[ii].round().astype(np.int32)
        # color = (int(colors[ii,0]),int(colors[ii,1]),int(colors[ii,2]))
        color = fixed_color
        if visibs[ii] > 0.5:
            thickness = -1 # filled in
        else:
            thickness = 1 # hollow
        if confs[ii] > conf_thr:
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
    B,T,C,H,W = rgbs.shape
    fn = args.file
    assert C == 3
    device = rgbs.device
    assert(B==1)

    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

    torch.cuda.empty_cache()
    print('starting forward...')
    f_start_time = time.time()

    flows_e, visconf_maps_e, _, _ = \
        model(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None, is_training=False)
    traj_maps_e = flows_e + grid_xy # B,Tf,2,H,W
    if args.query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = \
            model(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
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

    # subsample to make the vis more readable
    rate = args.subsample_rate
    trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2

    print('trajs_e', trajs_e.shape)
    # add a saving track feature
    # Get shape
    B, T, N, _ = trajs_e.shape
    assert B == 1

    # Convert to numpy
    tracks = trajs_e[0].cpu().numpy()  # (T, N, 2)
    visibility = visconfs_e[0,:,:,0].cpu().numpy() > 0.5  # (T, N) -> boolean mask

    # Compute actual scale factors used for height and width
    scale_h = H / orig_H  # scaled height / original height
    scale_w = W / orig_W  # scaled width / original width

    # Apply inverse scaling to x (width) and y (height) separately
    tracks_rescaled = tracks.copy()
    tracks_rescaled[..., 0] = tracks[..., 0] / scale_w  # x-coords
    tracks_rescaled[..., 1] = tracks[..., 1] / scale_h  # y-coords
    dim_rescaled = (orig_H, orig_W)

    # Stereo4d
    save_dir = os.path.join("2d-track/stereo4d", fn)
    # davis
    # save_dir = os.path.join("2d-track/davis", fn)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{fn}.npz")

    # Save to npz
    np.savez(save_path, tracks=tracks_rescaled, visibility=visibility, dim=dim_rescaled)
    print(f"Saved to {save_path}")
    print(tracks_rescaled.shape)
    print(visibility.shape)
    print(dim_rescaled)

    xy0 = trajs_e[0,0].cpu().numpy()
    colors = utils.improc.get_2d_colors(xy0, H, W)

    # sort according to velocity, so that moving points are drawn last
    vels = trajs_e[0,1:].detach().cpu().numpy() - trajs_e[0,:-1].detach().cpu().numpy() # T-1,N,2
    vels = np.linalg.norm(vels, axis=-1).mean(axis=0)
    inds = np.argsort(vels)

    # fn = args.mp4_path.split('/')[-1].split('.')[0]
    # Stereo4d
    rgb_out_f = f'./output-videos/stereo4d/{fn}/pt_vis_%s_rate%d_q%d.mp4' % (fn, rate, args.query_frame)
    print('rgb_out_f', rgb_out_f)
    temp_dir = f'./output-videos/stereo4d/{fn}/temp_pt_vis_%s_rate%d_q%d' % (fn, rate, args.query_frame)
    utils.basic.mkdir(temp_dir)
    vis = []
    for ti in range(T):
        pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
                          trajs_e[0,ti].detach().cpu().numpy(),
                          visconfs_e[0,ti,:,0].detach().cpu().numpy(),
                          visconfs_e[0,ti,:,1].detach().cpu().numpy(),
                          colors=colors,
                          radius=4,
                          inds=inds)
        vis.append(pt_vis)
    for ti in range(T):
        temp_out_f = '%s/%03d.png' % (temp_dir, ti)
        im = PIL.Image.fromarray(vis[ti])
        im.save(temp_out_f, "PNG", subsampling=0, quality=100)
    # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 30 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))

    import subprocess
    import imageio_ffmpeg

    # Get the path to the bundled ffmpeg executable
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # Build the command
    cmd = [
        ffmpeg_path, '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'image2', '-framerate', '30',
        '-pattern_type', 'glob', '-i', f'{temp_dir}/*.png',
        '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
        rgb_out_f
    ]

    # Run the command
    subprocess.run(cmd, check=True)
    
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
    # davis
    # mp4_path = f"/path/to/data/DAVIS/JPEGImages/480p/videos/{args.file}.mp4"

    # stereo4d
    mp4_path = f"/path/to/data/stereo4d/test_mp4s/{args.file}-left_rectified.mp4"
    rgbs = read_mp4(mp4_path)
    print('rgbs[0]', rgbs[0].shape)
    H,W = rgbs[0].shape[:2]
    orig_H,orig_W = rgbs[0].shape[:2]
    
    # shorten & shrink the video, in case the gpu is small
    rgbs = rgbs[:400]
    # Here!
    HH = 512
    scale = min(HH/H, HH/W)
    H, W = int(H*scale), int(W*scale)
    H, W = H//8 * 8, W//8 * 8 # make it divisible by 8
    rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    print('rgbs[0]', rgbs[0].shape)

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
    # parser.add_argument("--mp4_path", type=str, default='./demo_video/monkey.mp4') # input video
    parser.add_argument("--file", type=str, default='bear') # input video
    parser.add_argument("--query_frame", type=int, default=0) # which frame to track from
    parser.add_argument("--inference_iters", type=int, default=4) # number of inference steps per forward
    parser.add_argument("--window_len", type=int, default=16) # model hyperparam
    parser.add_argument("--subsample_rate", type=int, default=16) # vis hyp
    parser.add_argument("--mixed_precision", action='store_true', default=False)
    parser.add_argument("--mask", action='store_true', default=False)
    args = parser.parse_args()

    from nets.alltracker import Net; model = Net(args.window_len)
    count_parameters(model)

    run(model, args)
    
