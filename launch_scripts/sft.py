"""Recipe training entry point for MolmoMotion.

Trains a `trajectory_3d_*` mixture starting from a Molmo2 checkpoint:

    torchrun --nproc-per-node=8 launch_scripts/sft.py \\
        /path/to/Molmo2-4B-Pretrain \\
        trajectory_3d_human_p8_h3_f8 \\
        --save_folder=checkpoints/MolmoMotion-Stage1 \\
        ...

See the Training section of the README for the released Stage-1/Stage-2
recipes and the dataset-name grammar (`molmo_motion/data/get_dataset.py`).
"""

import argparse
import dataclasses
import os
from os.path import join
from typing import List

from omegaconf import OmegaConf, omegaconf

from molmo_motion.data.data_loader import WeightedDataset, KwargsMixture, DataLoaderConfig
from molmo_motion.data.dynamic_packer import PackingConfig
from molmo_motion.models.molmo.molmo import MolmoConfig
from molmo_motion.models.molmo2.molmo2 import Molmo2Config
from molmo_motion.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from molmo_motion.preprocessing.multicrop_preprocessor import MultiCropConfig
from molmo_motion.preprocessing.video_preprocessor import VideoPreprocessorConfig
from molmo_motion.torch_util import get_world_size
from molmo_motion.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from molmo_motion.train.run_trainer import run_trainer
from molmo_motion.train.trainer_config import FSDPConfig, BatchDivisor, SpeedMonitorConfig, TrainConfig, \
    WandbConfig, CompilerConfig
from molmo_motion.util import prepare_torchrun_environment, select_checkpoint, clean_opt


def get_model(checkpoint, model):
    model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")
    video_preprocessor_cfg = VideoPreprocessorConfig(
        pooling_h=3,
        pooling_w=3,
        time_mode="per-frame-compact",
        max_frames=128,
        time_sampling=True,
        loading_method="torchcodec_exact",
        frame_sample_mode="uniform_last_frame",
        max_fps=[2],
    )
    if isinstance(model_cfg.mm_preprocessor, MultiCropConfig):
        # Might be starting from a `Molmo` not a Molmo2` model
        kwargs = {field.name: getattr(model_cfg.mm_preprocessor, field.name) for field in dataclasses.fields(MultiCropConfig)}
        image_preprocessor_cfg = MultiCropConfig(**kwargs)
    else:
        image_preprocessor_cfg = model_cfg.mm_preprocessor.image

    # Choose config class based on model type
    config_kwargs = dict(
        llm=model_cfg.llm,
        vision_backbone=model_cfg.vision_backbone,
        data_formatter=model_cfg.data_formatter,
        mm_preprocessor=Molmo2PreprocessorConfig(
            video=video_preprocessor_cfg,
            image=image_preprocessor_cfg,
        ),
        bi_directional_attn=model_cfg.bi_directional_attn,
    )
    if model == "video_olmo_trajectory":
        from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig
        model_cfg = Molmo2TrajectoryConfig(**config_kwargs)
    else:
        model_cfg = Molmo2Config(**config_kwargs)

    # Fine-tuning settings
    model_cfg.vision_backbone.pooling_attention_mask = True
    model_cfg.data_formatter.pointing_format = "html-v2"
    model_cfg.mm_preprocessor.video.max_subtitle_tokens = None
    model_cfg.data_formatter.p_multi_point_all_image = 0.5
    model_cfg.data_formatter.p_choice_content_in_mc = 1.0

    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model_v2"
    model_cfg.data_formatter.message_format = "qwen3"
    model_cfg.data_formatter.system_prompt = "demo_or_style_v2"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments_root_tokens"

    # Multi-image settings
    model_cfg.mm_preprocessor.image.max_multi_image_crops = 8
    model_cfg.mm_preprocessor.image.max_images = 5

    # Good enough for 128 frames
    model_cfg.llm.max_sequence_length = 4096*4

    # Reduce shared memory requirements
    model_cfg.vision_backbone.normalize_on_gpu = True

    return model_cfg


def get_training_mixture(name):
    if not name.startswith("trajectory_3d"):
        raise NotImplementedError(
            f"Mixture {name!r} is not part of the public release. "
            f"Only 'trajectory_3d_*' mixtures are supported — see the "
            f"dataset-name grammar in molmo_motion/data/get_dataset.py."
        )
    # Multi-dataset 3D trajectory: trajectory_3d, trajectory_3d_p16_h3_f8, etc.
    # The suffix after "trajectory_3d" is parsed by get_dataset_by_name,
    # e.g. "_human_p8_h3_f8" or "_egodex_droid_p8_h3_f8".
    suffix = name[len("trajectory_3d"):]
    training_mixture = [
        ["trajectory_3d", [f"trajectory_3d{suffix}"], 1.0],
    ]
    root_size_mixture: List[KwargsMixture] = []
    for group_name, submixture, rate in training_mixture:
        submixture = [WeightedDataset.build(x) for x in submixture]
        root_size_mixture.append(KwargsMixture(rate, submixture, group_name))
    return root_size_mixture


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Train a MolmoMotion model")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("mixture", help="trajectory_3d_* mixture name, e.g. trajectory_3d_human_p8_h3_f8")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", default="video")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--device_batch_size", default=2, type=int)
    parser.add_argument("--max_loss_examples", default=2048, type=int)
    parser.add_argument("--max_inf_eval_examples", default=1280, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--cp_degree", default=1, type=int)
    args, other_args = parser.parse_known_args()

    training_mixture = get_training_mixture(args.mixture)
    suffix = args.mixture[len("trajectory_3d"):]
    loss_eval_tasks = [f"trajectory_3d_test{suffix}"]
    seq_len = args.seq_len

    checkpoint = select_checkpoint(args.checkpoint)
    model_cfg = get_model(checkpoint, args.model)

    if args.debug:
        checkpoint = None

        # Use a dummy model, but one still based on the input checkpoint
        model_cfg.llm.init_path = None
        model_cfg.llm.n_layers = 1
        if hasattr(model_cfg, "vision_backbone"):
            vit = model_cfg.vision_backbone.vit
            model_cfg.vision_backbone.vit_layers = [-1, -2]
        else:
            model_cfg.image_layers = [0]
            model_cfg.connector.vit_layers = [-1, -2]
            vit = model_cfg.vit
        vit.init_path = None
        vit.image_num_layers = 2
        args.num_workers = 2
        args.prefetch_factor = 2

    num_workers = args.num_workers

    loss_evaluations = []
    for task in loss_eval_tasks:
        from molmo_motion.eval.eval_utils import get_evaluation
        evaluation = get_evaluation(
            task,
            seq_len=seq_len,
            for_inference=False,
            device_batch_size=args.device_batch_size*2,
            max_examples=args.max_loss_examples,
            num_workers=num_workers,
        )
        evaluation.data.max_text_seq_len = None
        evaluation.data.pad = "to_max"
        evaluation.data.persistent_workers = True
        evaluation.data.prefetch_factor = args.prefetch_factor
        loss_evaluations.append(evaluation)

    # Inference evaluator on the held-out trajectory test slice
    from molmo_motion.eval.inf_evaluator import InfDatasetEvaluatorConfig, EvaluatorConfig, SamplingConfig

    traj3d_eval = InfDatasetEvaluatorConfig(
        label="trajectory_3d_test",
        data=DataLoaderConfig(
            dataset=f"trajectory_3d_test{suffix}",
            split="test",
            seed=691203,
            pad=None,
            sequence_length=None,
            max_text_seq_len=512,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        ),
        evaluator=EvaluatorConfig(egodex_3d_eval=True),
        max_new_tokens=4096,
        device_batch_size=args.device_batch_size,
        sampling=SamplingConfig(temperature=0.0),
        max_examples=args.max_inf_eval_examples,
        console_log_interval=1,
    )
    evaluations = [traj3d_eval]

    log_interval = 1 if args.debug else 20

    if args.debug or os.environ.get("WANDB_MODE", "").lower() == "disabled":
        wandb_cfg = None
    else:
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")
        if not wandb_project or not wandb_entity:
            raise ValueError(
                "Training logs to Weights & Biases: export WANDB_PROJECT and "
                "WANDB_ENTITY before launching (or export WANDB_MODE=disabled to "
                "turn logging off). See the Training section of the README."
            )
        wandb_cfg = WandbConfig(
            name="${run_name}",
            project=wandb_project,
            group=None,
            entity=wandb_entity,
            log_interval=log_interval,
            allow_resume=False,
            finish_on_sigterm=True,
        )

    cfg = TrainConfig(
        run_name="multitask_train",
        save_folder=omegaconf.MISSING,
        seed=6198,
        dry_run=False,

        wandb=wandb_cfg,
        compile=CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=True,
        data=DataLoaderConfig(
            kwargs_mixture=training_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=seq_len,
            max_text_seq_len=None,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            seed=50189,
            packing=PackingConfig(buffer_size=48, image_weight=30, shortcut_max_len_images=False,
                                  cp_world_size=args.cp_degree)
        ),
        ft_connector=True,
        ft_llm=not args.debug,
        ft_vit=not args.debug,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
            frame_selector_learning_rate=1e-4,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            frame_selector_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(fsdp2=True),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=2000,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=get_world_size() if args.debug else 128,
        device_train_microbatch_size=args.device_batch_size,
        time_limit=None,
        max_duration=300000,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        inf_evaluators=evaluations,
        evaluators=loss_evaluations,
        inf_eval_interval=-1,
        eval_interval=-1,
        save_final_unsharded_checkpoint=False,
        save_final_optim=False,
        response_logits_only=True,
    )

    cfg.parallelism.context_parallel_config.degree = args.cp_degree

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    conf = OmegaConf.to_object(conf)

    if conf.parallelism.context_parallel_config.degree > 1:
        conf.model.cp_enabled = True
        conf.compile = None  # compilation is not supported with context parallelism

    run_trainer(conf)


if __name__ == '__main__':
    main()
