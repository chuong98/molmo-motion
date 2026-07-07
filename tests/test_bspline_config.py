"""Task 6: bspline_n_ctrl persists through the public config and the internal
Molmo2TrajectoryConfig, so a trained checkpoint records the answer mode and the
public predict_trajectory / processor can pick the control-point decode + prompt.
"""

import dataclasses

from molmo_motion.public_config import MolmoMotionConfig


def test_public_config_persists_bspline():
    c = MolmoMotionConfig(num_points=8, history_size=3, future_size=32, bspline_n_ctrl=10)
    d = c.to_dict()
    assert d["bspline_n_ctrl"] == 10
    assert MolmoMotionConfig.from_dict(d).bspline_n_ctrl == 10


def test_public_config_defaults_frame_mode():
    assert MolmoMotionConfig().bspline_n_ctrl == 0


def test_internal_config_has_bspline_field():
    from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig

    # Dataclass field exists with a frame-mode default.
    names = {f.name for f in dataclasses.fields(Molmo2TrajectoryConfig)}
    assert "bspline_n_ctrl" in names
