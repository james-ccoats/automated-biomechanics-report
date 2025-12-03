import ezc3d
import pandas as pd
import matplotlib.pyplot as plt

file = '/Users/jamesccoats/Downloads/000005_000006_72_209_R_014_670.c3d'

c= ezc3d.c3d(file)

num_markets = (c['header']['points']['size'])

point_frame_rate = (c['header']['points']['frame_rate'])
point_first_frame = (c['header']['points']['first_frame'])
point_last_frame = (c['header']['points']['last_frame'])

analog_frame_rate = (c['header']['analogs']['frame_rate'])
analog_first_frame = (c['header']['analogs']['first_frame'])
analog_last_frame = (c['header']['analogs']['last_frame'])

print(point_frame_rate, point_first_frame, point_last_frame)
print(analog_frame_rate, analog_first_frame, analog_last_frame)

points = c['data']['points']
analogs = c['data']['analogs']

labels = c["parameters"]["POINT"]["LABELS"]["value"]

marker5_index = [i for i, label in enumerate(labels) if label == "Marker5"][0]

fig = plt.figure(figsize=(14, 5))
select_frames = [100, 333, 370, 400]
approx_swing_init_frame = 0
for i, frame in enumerate(select_frames):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    x = points[0, :, frame]
    y = points[1, :, frame]
    z = points[2, :, frame]
    marker_points = ax.scatter(x, y, z, alpha=0.75, color="royalblue")
    barrel_points = ax.scatter(
        xs=points[0, marker5_index, approx_swing_init_frame:frame],
        ys=points[1, marker5_index, approx_swing_init_frame:frame],
        zs=points[2, marker5_index, approx_swing_init_frame:frame],
        color="darkgoldenrod",
        alpha=0.5
    )
    ax.set_title(f"Frame {frame} of 768")
    ax.axis('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=30, azim=-55, roll=0)

fig.legend((marker_points, barrel_points), ("Body Markers", "Barrel"), bbox_to_anchor=(0, 0.5, 0.25, 0.4));





