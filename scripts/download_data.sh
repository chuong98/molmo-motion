MOLMO_MOTION_1M_ROOT=/data/molmo_motion_1m
POINTMOTIONBENCH_ROOT=/data/point_motion_bench
# Training corpus.
hf download allenai/molmo-motion-1m \
    --repo-type dataset --local-dir $MOLMO_MOTION_1M_ROOT

# Evaluation benchmark — only needed for `launch_scripts/eval_pointmotionbench.py`.
# hf download allenai/PointMotionBench \
#     --repo-type dataset --local-dir $POINTMOTIONBENCH_ROOT

# hf download allenai/MolmoMotion-4B-H3-F30 \
#     --local-dir checkpoints/MolmoMotion-4B-H3-F30