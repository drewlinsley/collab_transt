import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pylab
import imageio
from imageio import mimsave
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import patches


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile, detections, output_dir, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param)

    # Load a detections file
    csv = pd.read_csv(detections)
    video_id = int(videofile.split(os.path.sep)[-1].split("_")[0])
    video_view = videofile.split(os.path.sep)[-1].split("_")[-1].split(".")[0]
    names = csv.video_frame

    # Trim by views first
    views = [x.split("_")[-2] for x in names.values]
    view_mask = np.asarray(views) == np.asarray(video_view)
    csv = csv[view_mask]

    # Now trim by names
    names = csv.video_frame.values
    views = [x.split("_")[-2] for x in names]
    # names = [1 for name in names if name.split("_")[1] == 1 else 0]
    ids, idxs = np.unique([int(name.split("_")[0]) for name in names], return_index=True)

    n = np.where(ids == video_id)[0]  # np.where(np.logical_and(ids == video_id, np.asarray(views) == np.asarray(video_view)))
    idx = idxs[n]
    name = names[idx]
    mask = names == name
    hcsv = csv[mask].values
    hcsv = hcsv[:, 1: -1]
    hcsv = hcsv[:, [0, 2, 1, 3]]
    output_file = output_dir + name.tolist()[0]

    # tracks = []
    import pdb;pdb.set_trace()
    tracks = tracker.run_nfl(videofilepath=videofile, optional_box=hcsv, debug=debug, save_results=False)
    # for helmet in tqdm(hcsv, total=len(hcsv)):
    #     # Now run tracker for each detection
    #     helmet = tuple(helmet)
    #     track = tracker.run_nfl(videofilepath=videofile, optional_box=helmet, debug=debug, save_results=False)
    #     tracks.append(track)
    tracks = np.asarray(tracks)  # helmet X time X coord
    tracks = tracks.transpose((1, 0, 2))  # tracks = tracks[:, [1, 0, 2]]  # Multiproc changes column ordering
    np.save(output_file, tracks)

    # Plot a video for debugging
    cmap = get_cmap("hsv", tracks.shape[0] * 2)
    colors = np.random.rand(tracks.shape[0], 3)
    vid = imageio.get_reader(videofile,  'ffmpeg')
    images = []
    for t in range(tracks.shape[1]):
        frame = vid.get_data(t)
        # Plot each helmet
        fig, ax = plt.subplots(1, 1)
        plt.imshow(frame)
        for hid in range(tracks.shape[0]):  # , color in enumerate(cmap):
            helmet = tracks[hid, t]
            rect = patches.Rectangle((helmet[0], helmet[1]), helmet[2], helmet[3], linewidth=1, edgecolor=colors[hid], facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.axis("off")
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image_from_plot)
        plt.close(fig)
    mimsave(output_file + ".gif", images)



def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, help='Name of pcarameter file.')
    parser.add_argument('--videofile', type=str, help='path to a video file.')
    parser.add_argument('--detections', type=str, help='path to a helmet detections csv.')
    parser.add_argument('--output_dir', type=str, help='path to save tracks.')

    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()
    run_video(args.tracker_name, args.tracker_param, args.videofile, args.detections, args.output_dir, args.optional_box, args.debug, args.save_results)

if __name__ == '__main__':
    main()

