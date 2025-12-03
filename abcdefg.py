import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

def animate_skeleton_hitting(c3d_file, frame_step=4, marker_size=20, line_width=2, bat_color="saddlebrown"):
    """
    Animate a 3D skeleton from a C3D file with bat highlighted.

    Parameters
    ----------
    c3d_file : str
        Path to the C3D file.
    frame_step : int
        Step size to sample frames.
    marker_size : int
        Size of scatter points for markers.
    line_width : int
        Width of lines for skeleton and bat.
    bat_color : str
        Color for bat lines.
    """
    # Load C3D
    c = ezc3d.c3d(c3d_file)
    points = c['data']['points']
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    marker_dict = {label: i for i, label in enumerate(labels)}

    point_first_frame = c['header']['points']['first_frame']
    point_last_frame = c['header']['points']['last_frame']

    frame_points = np.arange(point_first_frame, point_last_frame, frame_step)

    # Skeleton & bat connections
    skeleton_connections = [
        ("LFIN", "LELB"), ("LELB", "LSHO"), ("LSHO", "C7"), ("C7", "RSHO"),
        ("RSHO", "RELB"), ("RELB", "RFIN"), ("C7", "T10"), ("T10", "LIC"),
        ("LIC", "LKNEE"), ("LKNEE", "LANK"), ("T10", "RIC"), ("RIC", "RKNEE"), ("RKNEE", "RANK")
    ]
    bat_connections = [("MARKER1", "MARKER2"), ("MARKER2", "MARKER3"), ("MARKER1", "MARKER3")]

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([0, 2000])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=30, azim=-55)

    # Prepare scatter & line objects
    scatter = ax.scatter([], [], [], s=marker_size, color="royalblue", alpha=0.6)
    skeleton_lines = [ax.plot([], [], [], color="darkorange", lw=line_width)[0] for _ in skeleton_connections]
    bat_lines = [ax.plot([], [], [], color=bat_color, lw=line_width+1)[0] for _ in bat_connections]

    # Update function for animation
    def update(frame_idx):
        frame = frame_points[frame_idx]
        x = points[0, :, frame]
        y = points[1, :, frame]
        z = points[2, :, frame]
        scatter._offsets3d = (x, y, z)

        for i, (p1, p2) in enumerate(skeleton_connections):
            if p1 in marker_dict and p2 in marker_dict:
                skeleton_lines[i].set_data([points[0, marker_dict[p1], frame], points[0, marker_dict[p2], frame]],
                                           [points[1, marker_dict[p1], frame], points[1, marker_dict[p2], frame]])
                skeleton_lines[i].set_3d_properties([points[2, marker_dict[p1], frame], points[2, marker_dict[p2], frame]])

        for i, (p1, p2) in enumerate(bat_connections):
            if p1 in marker_dict and p2 in marker_dict:
                bat_lines[i].set_data([points[0, marker_dict[p1], frame], points[0, marker_dict[p2], frame]],
                                      [points[1, marker_dict[p1], frame], points[1, marker_dict[p2], frame]])
                bat_lines[i].set_3d_properties([points[2, marker_dict[p1], frame], points[2, marker_dict[p2], frame]])

        return [scatter] + skeleton_lines + bat_lines

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frame_points), interval=50, blit=False)
    
    # Add legend
    body_line = mlines.Line2D([], [], color="darkorange", linewidth=line_width, label="Body")
    bat_line = mlines.Line2D([], [], color=bat_color, linewidth=line_width+1, label="Bat")
    ax.legend(handles=[body_line, bat_line], bbox_to_anchor=(0, 0.5, 1, 0.5))

    plt.show()
    return ani

animate_skeleton_hitting('/Users/jamesccoats/Downloads/000044_000265_67_171_L_001_929.c3d', frame_step=50)