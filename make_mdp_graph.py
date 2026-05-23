"""Generate simple visualizations for the CPU scheduling MDP state space."""

from __future__ import annotations

import json
from pathlib import Path

from cpu_scheduling_env import CPUSchedulingEnv


OUT_DIR = Path("submission_assets")
OUT_FILE = OUT_DIR / "full_mdp_graph.svg"
OUT_3D_FILE = OUT_DIR / "full_mdp_graph_3d.html"


MODE_COLORS = {
    CPUSchedulingEnv.MODE_NONE: "#9ca3af",
    CPUSchedulingEnv.MODE_SHORT: "#5b8bd9",
    CPUSchedulingEnv.MODE_MEDIUM: "#78a96b",
    CPUSchedulingEnv.MODE_LONG: "#d89c52",
}

MODE_LABELS = {
    CPUSchedulingEnv.MODE_NONE: "Idle",
    CPUSchedulingEnv.MODE_SHORT: "Short",
    CPUSchedulingEnv.MODE_MEDIUM: "Medium",
    CPUSchedulingEnv.MODE_LONG: "Long",
}


def state_position(state):
    q_short, q_medium, q_long, mode = state
    left = 70
    top = 70
    cell_w = 110
    cell_h = 110

    base_x = left + (q_short + q_long * 3) * cell_w
    base_y = top + q_medium * cell_h
    offsets = {
        CPUSchedulingEnv.MODE_NONE: (22, 24),
        CPUSchedulingEnv.MODE_SHORT: (78, 24),
        CPUSchedulingEnv.MODE_MEDIUM: (22, 78),
        CPUSchedulingEnv.MODE_LONG: (78, 78),
    }
    dx, dy = offsets[mode]
    return base_x + dx, base_y + dy


def build_svg(env):
    width = 1100
    height = 470
    positions = {state: state_position(state) for state in env.states}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for q_long in range(3):
        for q_medium in range(3):
            for q_short in range(3):
                x = 70 + (q_short + q_long * 3) * 110
                y = 70 + q_medium * 110
                parts.append(
                    f'<rect x="{x}" y="{y}" width="100" height="100" rx="8" '
                    'fill="#f8fafc" stroke="#d5dce6" stroke-width="1"/>'
                )

    for state in env.states:
        x, y = positions[state]
        mode = state[3]
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="11" fill="{MODE_COLORS[mode]}" '
            'stroke="#222" stroke-width="1.1"/>'
        )

    legend_x = 70
    legend_y = 430
    for idx, mode in enumerate(
        [
            CPUSchedulingEnv.MODE_NONE,
            CPUSchedulingEnv.MODE_SHORT,
            CPUSchedulingEnv.MODE_MEDIUM,
            CPUSchedulingEnv.MODE_LONG,
        ]
    ):
        x = legend_x + idx * 150
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="9" fill="{MODE_COLORS[mode]}" stroke="#222"/>')
        parts.append(
            f'<text x="{x + 18}" y="{legend_y + 5}" font-family="Arial" font-size="16">{MODE_LABELS[mode]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def build_3d_html(env):
    nodes = []
    for idx, state in enumerate(env.states):
        q_short, q_medium, q_long, mode = state
        mode_offsets = {
            CPUSchedulingEnv.MODE_NONE: (-0.12, -0.12, 0.00),
            CPUSchedulingEnv.MODE_SHORT: (0.12, -0.12, 0.00),
            CPUSchedulingEnv.MODE_MEDIUM: (-0.12, 0.12, 0.00),
            CPUSchedulingEnv.MODE_LONG: (0.12, 0.12, 0.00),
        }
        dx, dy, dz = mode_offsets[mode]
        nodes.append(
            {
                "id": idx,
                "x": q_short + dx,
                "y": q_medium + dy,
                "z": q_long + dz,
                "mode": mode,
                "label": f"({q_short},{q_medium},{q_long},{MODE_LABELS[mode]})",
            }
        )

    state_to_id = {state: idx for idx, state in enumerate(env.states)}
    edge_set = set()
    for state in env.states:
        src = state_to_id[state]
        for action in env.actions:
            for next_state in env.states:
                if env.transition_prob(next_state, state, action) > 0:
                    dst = state_to_id[next_state]
                    if src != dst:
                        edge_set.add((src, dst))
    edges = [{"source": src, "target": dst} for src, dst in sorted(edge_set)]

    colors = {str(mode): color for mode, color in MODE_COLORS.items()}
    labels = {str(mode): label for mode, label in MODE_LABELS.items()}
    nodes_json = json.dumps(nodes, separators=(",", ":"))
    edges_json = json.dumps(edges, separators=(",", ":"))
    colors_json = json.dumps(colors, separators=(",", ":"))
    labels_json = json.dumps(labels, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CPU Scheduling MDP 3D Graph</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      font-family: Arial, sans-serif;
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
      cursor: grab;
    }}
    canvas:active {{
      cursor: grabbing;
    }}
    .legend {{
      position: fixed;
      left: 24px;
      bottom: 22px;
      display: flex;
      gap: 24px;
      align-items: center;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid #ddd;
      border-radius: 8px;
      font-size: 15px;
    }}
    .item {{
      display: flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
    }}
    .dot {{
      width: 13px;
      height: 13px;
      border-radius: 50%;
      border: 1px solid #222;
    }}
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <div class="legend" id="legend"></div>
  <script>
    const nodes = {nodes_json};
    const edges = {edges_json};
    const modeColors = {colors_json};
    const modeLabels = {labels_json};

    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const legend = document.getElementById("legend");

    for (const mode of ["0", "1", "2", "3"]) {{
      const item = document.createElement("div");
      item.className = "item";
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = modeColors[mode];
      const text = document.createElement("span");
      text.textContent = modeLabels[mode];
      item.append(dot, text);
      legend.appendChild(item);
    }}

    let width = 0;
    let height = 0;
    let yaw = -0.72;
    let pitch = 0.58;
    let zoom = 185;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const center = {{ x: 1, y: 1, z: 1 }};

    function resize() {{
      const scale = window.devicePixelRatio || 1;
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * scale);
      canvas.height = Math.floor(height * scale);
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      draw();
    }}

    function hexToRgba(hex, alpha) {{
      const n = parseInt(hex.slice(1), 16);
      const r = (n >> 16) & 255;
      const g = (n >> 8) & 255;
      const b = n & 255;
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }}

    function project(point) {{
      let x = point.x - center.x;
      let y = point.y - center.y;
      let z = point.z - center.z;

      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);

      const x1 = x * cy - z * sy;
      const z1 = x * sy + z * cy;
      const y1 = y * cp - z1 * sp;
      const z2 = y * sp + z1 * cp;

      const perspective = 1 / (1 + z2 * 0.22);
      return {{
        x: width / 2 + x1 * zoom * perspective,
        y: height / 2 - y1 * zoom * perspective,
        z: z2,
        scale: perspective,
      }};
    }}

    function drawAxis() {{
      const origin = project({{x: -0.25, y: -0.25, z: -0.25}});
      const axes = [
        [{{x: 2.45, y: -0.25, z: -0.25}}, "qS"],
        [{{x: -0.25, y: 2.45, z: -0.25}}, "qM"],
        [{{x: -0.25, y: -0.25, z: 2.45}}, "qL"],
      ];
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = "rgba(60, 60, 60, 0.55)";
      ctx.fillStyle = "rgba(40, 40, 40, 0.75)";
      ctx.font = "16px Arial";
      for (const [endPoint, label] of axes) {{
        const end = project(endPoint);
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
        ctx.fillText(label, end.x + 8, end.y + 4);
      }}
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, width, height);

      const projected = nodes.map(project);
      drawAxis();

      ctx.lineWidth = 1.1;
      for (const edge of edges) {{
        const a = projected[edge.source];
        const b = projected[edge.target];
        ctx.strokeStyle = "rgba(55, 65, 80, 0.20)";
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }}

      const order = nodes.map((node, index) => [node, projected[index], index]).sort((a, b) => a[1].z - b[1].z);
      for (const [node, p] of order) {{
        const radius = 7.5 * p.scale + 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(modeColors[node.mode], 0.95);
        ctx.fill();
        ctx.strokeStyle = "rgba(25, 25, 25, 0.85)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }}
    }}

    canvas.addEventListener("mousedown", (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    }});
    window.addEventListener("mouseup", () => dragging = false);
    window.addEventListener("mousemove", (event) => {{
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.008;
      pitch += (event.clientY - lastY) * 0.008;
      pitch = Math.max(-1.25, Math.min(1.25, pitch));
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});
    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      zoom *= event.deltaY > 0 ? 0.92 : 1.08;
      zoom = Math.max(80, Math.min(420, zoom));
      draw();
    }}, {{ passive: false }});

    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""


def main():
    OUT_DIR.mkdir(exist_ok=True)
    env = CPUSchedulingEnv(max_queue=2, arrival_probs=(0.35, 0.25, 0.15), seed=2)
    OUT_FILE.write_text(build_svg(env), encoding="utf-8")
    OUT_3D_FILE.write_text(build_3d_html(env), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")
    print(f"Wrote {OUT_3D_FILE}")


if __name__ == "__main__":
    main()
