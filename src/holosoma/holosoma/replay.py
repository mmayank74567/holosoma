from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


@dataclass
class ReplayOutputConfig:
    output_name: str = "replay"
    """Stem of the output NPZ (saved to demo_trajectory/{exp_name}/{output_name}.npz)."""


def _write_mp4(frames: list, fps: float, out_path: Path) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        label = f"frame {i}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        x, y = w - tw - 10, th + 10
        cv2.rectangle(bgr, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(bgr, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        writer.write(bgr)
    writer.release()
    print(f"Saved video      → {out_path}")


def _save_torque_plot(joint_efforts: np.ndarray, dof_names: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_eff = np.abs(joint_efforts)  # (T, J)
    J = abs_eff.shape[1]
    names = list(dof_names)

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2,
        figsize=(18, max(6, J * 0.4)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    im = ax_heat.imshow(abs_eff.T, aspect="auto", cmap="inferno", origin="lower", interpolation="nearest")
    ax_heat.set_xlabel("Frame")
    ax_heat.set_title("Joint Effort Magnitude Over Time")
    ax_heat.set_yticks(range(J))
    ax_heat.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, ax=ax_heat, label="|Effort| (Nm)", fraction=0.046, pad=0.04)

    mean_eff = abs_eff.mean(axis=0)
    ax_bar.barh(range(J), mean_eff, color=plt.cm.inferno(mean_eff / (mean_eff.max() + 1e-6)))
    ax_bar.set_yticks(range(J))
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.set_xlabel("Mean |Effort| (Nm)")
    ax_bar.set_title("Average Effort")
    ax_bar.invert_yaxis()

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved torque plot → {out_path}")


def _save_link_force_plot(link_forces: np.ndarray, link_names: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # link_forces: (T, num_links, 6) — [:, :, 0:3] = force, [:, :, 3:6] = torque
    force_mag = np.linalg.norm(link_forces[:, :, 0:3], axis=-1)  # (T, num_links)
    torque_mag = np.linalg.norm(link_forces[:, :, 3:6], axis=-1)  # (T, num_links)
    L = len(link_names)

    fig, (ax_f, ax_t) = plt.subplots(
        2, 1,
        figsize=(16, max(8, L * 0.5)),
        sharex=True,
    )

    for ax, data, label, cmap in [
        (ax_f, force_mag.T,  "Force magnitude ||F|| (N)",  "plasma"),
        (ax_t, torque_mag.T, "Torque magnitude ||T|| (Nm)", "viridis"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap, origin="lower", interpolation="nearest")
        ax.set_yticks(range(L))
        ax.set_yticklabels(link_names, fontsize=7)
        ax.set_ylabel("Link")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    ax_t.set_xlabel("Frame")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved link force plot → {out_path}")


def replay(tyro_config: ExperimentConfig, output_name: str = "replay"):
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    sim = env.simulator
    physx_view = sim._robot.root_physx_view
    exp_name = tyro_config.env_class.split(".")[-2]
    out_dir = Path("demo_trajectory") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    motion_cmd = env.command_manager.get_state("motion_command")
    video_fps = float(motion_cmd.motion.fps)

    import omni.replicator.core as rep

    render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", do_array_copy=False)
    rgb_annotator.attach([render_product])

    link_names = [p.split("/")[-1] for p in physx_view.link_paths[0]]

    efforts_buf: list = []
    gravity_buf: list = []
    link_forces_buf: list = []
    dof_pos_buf: list = []
    frames_buf: list = []
    episode = 0

    done = False
    while not done:
        env.simulator.sim.step()
        # done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        done = env.step_visualize_motion(None)

        efforts_buf.append(physx_view.get_dof_projected_joint_forces()[0, sim.dof_ids].cpu().numpy())
        gravity_buf.append(physx_view.get_generalized_gravity_forces()[0, sim.dof_ids].cpu().numpy())
        link_forces_buf.append(physx_view.get_link_incoming_joint_force()[0].cpu().numpy())
        dof_pos_buf.append(sim.dof_pos[0].cpu().numpy())

        raw = rgb_annotator.get_data()
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        if isinstance(raw, np.ndarray) and raw.size > 0:
            frames_buf.append(raw[:, :, :3].copy())

        if done:
            stem = output_name if episode == 0 else f"{output_name}_ep{episode}"

            np.savez_compressed(
                str(out_dir / f"{stem}.npz"),
                joint_efforts=np.array(efforts_buf),
                gravity_forces=np.array(gravity_buf),
                link_incoming_forces=np.array(link_forces_buf),
                dof_pos=np.array(dof_pos_buf),
                dof_names=np.array(sim.dof_names),
                link_names=np.array(link_names),
            )
            print(f"Saved {len(efforts_buf)} steps → {out_dir / f'{stem}.npz'}")

            if frames_buf:
                _write_mp4(frames_buf, video_fps, out_dir / f"{stem}.mp4")

            _save_torque_plot(
                np.array(gravity_buf),
                np.array(sim.dof_names),
                out_dir / f"{stem}_torques.png",
            )

            _save_link_force_plot(
                np.array(link_forces_buf),
                link_names,
                out_dir / f"{stem}_link_forces.png",
            )

            efforts_buf.clear()
            gravity_buf.clear()
            link_forces_buf.clear()
            dof_pos_buf.clear()
            frames_buf.clear()
            episode += 1
            #env.simulator.sim.reset()


    close_simulation_app(simulation_app)


def main() -> None:
    output_cfg, remaining_args = tyro.cli(ReplayOutputConfig, return_unknown_args=True, add_help=False)
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG, args=remaining_args)
    replay(tyro_cfg, output_name=output_cfg.output_name)


if __name__ == "__main__":
    main()
