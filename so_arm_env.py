"""Ambiente Gymnasium para o robô **Standard Open Arm-100** (5 graus de liberdade).

Este arquivo disponibiliza um ambiente *mínimo, porém completo* para
aprendizado por reforço (RL) utilizando a descrição MJCF incluída neste
repositório. O ambiente pode ser usado de forma programática – por exemplo,
por um algoritmo de RL – ou simplesmente para visualização executando::

    python manual_control.py

Princípios de projeto
--------------------
* **Espaço de ação** – Posições desejadas (rad) para cada um dos seis
  atuadores controlados em posição descritos em *so_arm100.xml* (Rotation,
  Pitch, Elbow, Wrist_Pitch, Wrist_Roll e Jaw).
* **Espaço de observação** – Concatenação das posições articulares *qpos* com
  as velocidades *qvel* (tamanho ``model.nq + model.nv``).
* **Recompensa** – Nenhuma tarefa é definida aqui; o método :py:meth:`step`
  devolve sempre recompensa ``0.0``. Espera-se que o usuário crie um *wrapper*
  fornecendo o sinal de recompensa apropriado (por exemplo, alcançar um alvo
  ou agarrar um objeto).
* **Término do episódio** – O episódio continua indefinidamente, a menos que
  seja encerrado externamente (``terminated = truncated = False``).
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Tuple

import numpy as np

# O Gymnasium fornece a API padrão de ambientes.
import gymnasium as gym

# A importação do MuJoCo é adiada propositadamente para o *runtime* de forma
# que ferramentas de análise estática, testes automatizados etc. possam ser
# executados mesmo que a biblioteca não esteja instalada. A importação real
# acontece dentro de ``__init__``.


class SoArm100Env(gym.Env):
    """Env. Gymnasium que encapsula a descrição MJCF do SO-ARM100."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # ---------------------------------------------------------------------
    # Métodos auxiliares de construção
    # ---------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | os.PathLike | None = "scene.xml",
        frame_skip: int = 10,
        render_mode: str | None = "human",
    ) -> None:
        """Cria uma nova instância do ambiente.

        Parâmetros
        ----------
        model_path
            Caminho para o *scene.xml* que inclui o arquivo do robô
            *so_arm100.xml*. Se preferir carregar o robô diretamente, basta
            passar ``"so_arm100.xml"``.
        frame_skip
            Quantidade de sub-passos da simulação realizados a cada chamada de
            :py:meth:`step`.
        render_mode
            ``None`` para não renderizar, ``"human"`` para abrir a janela GLFW
            interativa ou ``"rgb_array"`` para obter um *frame* renderizado
            fora da tela.
        """

        super().__init__()

        # Importação em tempo de execução para evitar dependência dura no
        # momento de importação do módulo.
        try:
            import mujoco as _mj  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'mujoco' python package is required but not installed. "
                "Please follow the installation instructions at "
                "https://github.com/google-deepmind/mujoco and ensure MuJoCo ≥3.1.6." 
            ) from exc

        # Guardamos a referência para não precisar importar novamente a cada
        # uso. Também injetamos o módulo no espaço global para que os demais
        # métodos (definidos fora do __init__) o acessem sem repetição de
        # código.
        global mujoco  # noqa: PLW0603 – we deliberately inject the symbol.
        mujoco = _mj
        self._mujoco = _mj

        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode

        # -----------------------------------------------------------------
        # Carrega o modelo MJCF e cria o objeto MjData correspondente
        # -----------------------------------------------------------------
        model_path = pathlib.Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find MJCF model: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # -----------------------------------------------------------------
        # Define espaços de ação e observação
        # -----------------------------------------------------------------
        self._setup_spaces()

        # -----------------------------------------------------------------
        # Viewer opcional (instanciado sob demanda na primeira renderização)
        # -----------------------------------------------------------------
        self._viewer: "mujoco.viewer.Viewer | None" = None

        # Pré-calcula mapeamento *nome do atuador → índice* para conveniência.
        self._actuator_name_to_id = self._build_actuator_index()

        # Armazena uma cópia da posição *home* definida em *so_arm100.xml*.
        # Caso o keyframe seja removido do XML, caimos para vetores de zeros
        # sem quebrar a execução.
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

        # Reinicia qpos/qvel e também os comandos dos atuadores (ctrl).
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self._home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._home_ctrl

        # Garante que a dinâmica interna esteja consistente com o novo estado.
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()

        info: dict[str, Any] = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: np.ndarray):
        # Limita a ação ao intervalo permitido pelo atuador.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()

        # Sem recompensa ou condição de término específicas por padrão.
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
            return  # Sem renderização.

        if mode == "human":
            # Instancia o viewer (janela GLFW) apenas na primeira chamada.
            if self._viewer is None:
                from mujoco import viewer as mj_viewer

                # O *viewer passivo* não assume o contexto via ``with``; ele
                # permanece vivo até que ``.close()`` seja chamado.
                self._viewer = mj_viewer.launch_passive(self.model, self.data)

            # Mantém a janela sincronizada com a simulação.
            self._viewer.sync()
        elif mode == "rgb_array":
            # Renderização fora da tela – aloca na primeira chamada e reutiliza.
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model, 640, 480)

            self._renderer.update_scene(self.data, camera=None)
            return self._renderer.render().copy()
        else:
            raise NotImplementedError(f"Unsupported render mode '{mode}'")

    def close(self):  # type: ignore[override]
        # Fecha a janela do viewer caso ela tenha sido criada.
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Métodos auxiliares públicos
    # ------------------------------------------------------------------

    def _setup_spaces(self):
        """Cria :pyattr:`action_space` e :pyattr:`observation_space`."""

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = gym.spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        obs_high = np.inf * np.ones(self.model.nq + self.model.nv, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Retorna o vetor de observação atual (*qpos*‖*qvel*)."""

        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    # ------------------------------------------------------------------
    # Métodos auxiliares internos
    # ------------------------------------------------------------------

    def _build_actuator_index(self):
        """Gera um dicionário *nome do atuador → id do atuador*."""

        name_to_id: dict[str, int] = {}
        for i in range(self.model.nu):
            name_val = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name_val is None:
                continue

            # MuJoCo < 3.1.6 devolvia ``bytes`` enquanto ≥ 3.1.7 devolve ``str``.
            if isinstance(name_val, bytes):
                name_val = name_val.decode()

            name_to_id[str(name_val)] = i
        return name_to_id

    def _extract_home_keyframe(self):
        """Extrai *qpos*/*ctrl* do keyframe chamado "home" dentro do XML."""

        for key_id in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)

            if key_name is None:
                continue

            if isinstance(key_name, bytes):
                key_name = key_name.decode()

            if key_name == "home":
                # O keyframe armazena qpos (nq), qvel (nv) e ctrl (nu).
                self._home_qpos = self.model.key_qpos[key_id].copy()
                if self.model.nu:
                    self._home_ctrl = self.model.key_ctrl[key_id].copy()
                return

        # Caso não exista keyframe "home": vetores já iniciados em zero.
