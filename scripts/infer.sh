ROOT_DIR=/home/chuong/workspace/point_models
CAMERA_CALIB=$ROOT_DIR/molmo-motion/data_generation/astribot_camera_calib_params/astribot_calib_head_rgbd.json
OUTPUT_DIR=$ROOT_DIR/output/astribot_demo_depth
DEPTH_FRAME=$OUTPUT_DIR/depth/depth_000040.npz
RGB_FRAME=/home/chuong/workspace/demo_data/frames
python scripts/infer.py \
    --example egodex_ball_base  \
    --image $RGB_FRAME/0040.jpg  \
    --bbox 1 250 75 340 \
    --grid 4 4 \
    --depth $DEPTH_FRAME \
    --calib $CAMERA_CALIB --camera head_rgbd \
    --action "Move white ball to the right, far away from the 3D printed base" \
    --output $OUTPUT_DIR/pointcloud_0040/exp5 \