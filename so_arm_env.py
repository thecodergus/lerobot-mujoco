"""Gymnasium environment for the Standard Open Arm-100 (5-DoF) robot.

This file provides a minimal yet *complete* reinforcement-learning friendly
environment around the MJCF description delivered with this repository.  The
environment can be used programmatically (e.g. by an RL algorithm) or
executed manually from :pymod:`manual_control.py` where a small keyboard UI is
implemented.

Key design choices
------------------
* **Action space** – Desired joint positions (rad) for each of the six
  position-controlled actuators listed in *so_arm100.xml* (Rotation, Pitch,
  Elbow, Wrist_Pitch, Wrist_Roll, and Jaw).
* **Observation space** – Concatenation of joint positions *qpos* and joint
  velocities *qvel* (shape = ``(model.nq + model.nv,)``).
* **Reward** – This basic environment does *not* define a task specific
  reward; it returns ``0.0`` every step.  It is expected that downstream users
  wrap the environment and supply a task specific reward signal (for example,
  reaching or grasping).
* **Episode end** – Unless terminated externally, the episode continues
  indefinitely (``terminated = truncated = False``).

If you only need to *tele-operate* the arm, run::

    python manual_control.py

That script relies on this class under the hood.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Tuple

import numpy as np

# Gymnasium is used for the common environment API.
import gymnasium as gym

# We *intentionally* postpone importing MuJoCo until runtime so that tooling
# (static analysis, unit-tests that do not require the simulator, …) can work
# without MuJoCo installed.  The import happens inside ``__init__``.


class SoArm100Env(gym.Env):
    """Gymnasium environment wrapping the SO-ARM100 MJCF description."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | os.PathLike | None = "scene.xml",
        frame_skip: int = 10,
        render_mode: str | None = "human",
    ) -> None:
        """Create a new environment instance.

        Parameters
        ----------
        model_path
            Path to the *scene.xml* that includes the *so_arm100.xml* robot
            file.  If you prefer to load the robot directly, simply pass
            ``"so_arm100.xml"``.
        frame_skip
            Number of simulator sub-steps performed for every call to
            :py:meth:`step`.
        render_mode
            One of ``None``, ``"human"`` (interactive GLFW window) or
            ``"rgb_array"``.
        """

        super().__init__()

        # Runtime import to avoid hard dependency at *import* time.
        try:
            import mujoco as _mj  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'mujoco' python package is required but not installed. "
                "Please follow the installation instructions at "
                "https://github.com/google-deepmind/mujoco and ensure MuJoCo ≥3.1.6." 
            ) from exc

        # Store reference to avoid re-importing each time.
        # Expose module both as a *private* attribute and as a global symbol
        # so that the remaining class methods (defined outside __init__) can
        # access it without repeating the import boilerplate.
        global mujoco  # noqa: PLW0603 – we deliberately inject the symbol.
        mujoco = _mj
        self._mujoco = _mj

        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode

        # -----------------------------------------------------------------
        # Load MJCF model & corresponding MjData object
        # -----------------------------------------------------------------
        model_path = pathlib.Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find MJCF model: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # -----------------------------------------------------------------
        # Action & observation spaces
        # -----------------------------------------------------------------
        self._setup_spaces()

        # -----------------------------------------------------------------
        # Optional viewer (created lazily on first render)
        # -----------------------------------------------------------------
        self._viewer: "mujoco.viewer.Viewer | None" = None

        # Pre-compute mapping *actuator name → index* for convenience.
        self._actuator_name_to_id = self._build_actuator_index()

        # Save a copy of the *home* position defined in so_arm100.xml.  Should
        # the XML be modified and the keyframe removed, we gracefully fall
        # back to a zero vector.
        self._home_qpos = np.zeros(self.model.nq, dtype=np.float32)
        self._home_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        self._extract_home_keyframe()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Reset qpos/qvel as well as actuator targets (ctrl).
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self._home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._home_ctrl

        # Ensure internal forward dynamics are consistent with the new state.
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()

        info: dict[str, Any] = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: np.ndarray):
        # Clip the desired joint positions to the ctrl range.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()

        # By default no task specific reward or termination condition.
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self, mode: str | None = None):  # type: ignore[override]
        mode = mode or self.render_mode
        if mode is None:
            return  # No render.

        if mode == "human":
            # Lazy-instantiate viewer (GLFW window) once.
            if self._viewer is None:
                from mujoco import viewer as mj_viewer

                # Passive viewer does not take ownership of the context via
                # `with` and therefore stays alive until `.close()` is
                # called.
                self._viewer = mj_viewer.launch_passive(self.model, self.data)

            # Make sure the viewer is up to date.
            self._viewer.sync()
        elif mode == "rgb_array":
            # Off-screen render – allocate the first time and reuse.
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model, 640, 480)

            self._renderer.update_scene(self.data, camera=None)
            return self._renderer.render().copy()
        else:
            raise NotImplementedError(f"Unsupported render mode '{mode}'")

    def close(self):  # type: ignore[override]
        # Clean up viewer if it was created.
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _setup_spaces(self):
        """Populate :pyattr:`action_space` and :pyattr:`observation_space`."""

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = gym.spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        obs_high = np.inf * np.ones(self.model.nq + self.model.nv, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return the current observation vector (qpos‖qvel)."""

        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_actuator_index(self):
        """Return mapping *actuator name → actuator id*."""

        name_to_id: dict[str, int] = {}
        for i in range(self.model.nu):
            name_val = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name_val is None:
                continue

            # MuJoCo <3.1.6 returned ``bytes`` whereas 3.1.7+ returns ``str``.
            if isinstance(name_val, bytes):
                name_val = name_val.decode()

            name_to_id[str(name_val)] = i
        return name_to_id

    def _extract_home_keyframe(self):
        """Populate *home* qpos/ctrl from the XML *keyframe* named "home"."""

        for key_id in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)

            if key_name is None:
                continue

            if isinstance(key_name, bytes):
                key_name = key_name.decode()

            if key_name == "home":
                # Keyframe stores qpos (-> size nq), qvel (nv) and ctrl (nu).
                self._home_qpos = self.model.key_qpos[key_id].copy()
                if self.model.nu:
                    self._home_ctrl = self.model.key_ctrl[key_id].copy()
                return

        # Fallback: already initialised with zeros.
