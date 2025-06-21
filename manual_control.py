"""Demonstração minimalista do modelo SO-ARM100 no MuJoCo.

Ao executar::

    python manual_control.py

o script carrega o robô, abre a janela do *viewer* e mantém a simulação
rodando até que a janela seja fechada pelo usuário (por exemplo, clicando no
botão *X* ou usando *Esc* dentro do viewer padrão).  Nenhuma outra forma de
interação é implementada ─ o objetivo aqui é apenas visualizar o robô.
"""

from __future__ import annotations

# Terceiros -----------------------------------------------------------------
import mujoco
from mujoco import viewer as mj_viewer

# Locais ---------------------------------------------------------------------
from so_arm_env import SoArm100Env


def main() -> None:  # pragma: no cover – script interativo.
    """Executa a simulação e mostra a janela do MuJoCo viewer."""

    env = SoArm100Env(render_mode="human")

    # Viewer em modo passivo; precisamos cuidar do loop de simulação.
    with mj_viewer.launch(env.model, env.data) as v:
        while v.is_running():
            # Avança a simulação (sem controles adicionais).
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, env.data)

            # Actualiza a janela.
            v.sync()


if __name__ == "__main__":
    main()
