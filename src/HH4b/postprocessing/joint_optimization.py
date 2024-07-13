from __future__ import annotations

import pickle
from pathlib import Path

# # Create scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue')
# # Label a random subset of points to avoid clutter
# label_fraction = 0.1  # Fraction of points to label (adjust as needed)
# num_labels = int(label_fraction * len(labels))
# indices_to_label = random.sample(range(len(labels)), num_labels)
# for i in range(len(labels)):
#     plt.text(x[i], y[i], f"{labels[i]}", fontsize=9)
# # Add axis labels and title
# plt.xlabel('Figure of Merit VBF')
# plt.ylabel('Figure of Merit ggF')
# plt.title('Scatter Plot of Figures of Merit')
# plt.savefig('/home/users/annava/projects/HH4b/plots/PostProcess/24May16Joint/scatter_plot.png')
import matplotlib.pyplot as plt
import numpy as np


def is_dominated(point, other_points):
    for other in other_points:
        if all(o >= p for o, p in zip(other, point)) and any(o > p for o, p in zip(other, point)):
            return True
    return False


def is_dominated_bovers(point, other_points):
    for other in other_points:
        if all(o <= p for o, p in zip(other, point)) and any(o < p for o, p in zip(other, point)):
            return True
    return False


def compute_pareto_front(points, labels):
    pareto_front = []
    pareto_front_labels = []
    for point, label in zip(points, labels):
        if not is_dominated_bovers(point, points):  # is_dominated(point, points):
            pareto_front.append(point)
            pareto_front_labels.append(label)
    return pareto_front, pareto_front_labels


def graph_with_pareto_front(
    results, path="/home/users/annava/projects/HH4b/plots/PostProcess/24May16Joint"
):
    # Extract data for plotting
    # print(results.values())
    x = [
        2 * np.sqrt(value["nevents_bkg_vbf"]) / value["nevents_sig_vbf"]
        for value in results.values()
    ]
    y = [
        2 * np.sqrt(value["nevents_bkg_ggF"]) / value["nevents_sig_ggF"]
        for value in results.values()
    ]
    # x = [1/2*(value["figure_of_merit_vbf"]) for value in results.values()]
    # y = [1/2*(value["figure_of_merit_ggF"]) for value in results.values()]
    labels = list(results.keys())

    # Combine x and y into points
    points = list(zip(x, y))

    # Compute Pareto front
    pareto_front, pareto_front_labels = compute_pareto_front(points, labels)
    print("pareto front, labels:", pareto_front, pareto_front_labels)
    pareto_front.sort()

    # Extract Pareto front x and y
    pareto_x = [point[0] for point in pareto_front]
    pareto_y = [point[1] for point in pareto_front]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue")

    # Label each point with its coordinate
    # for i in range(len(labels)):
    #     plt.text(x[i], y[i], f"{labels[i]}", fontsize=9)

    # Plot Pareto front
    plt.plot(
        pareto_x,
        pareto_y,
        color="red",
        linestyle="-",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Pareto Front",
    )

    opt_vbf_point = min(points, key=lambda p: p[0])
    opt_ggf_point = min(points, key=lambda p: p[1])
    opt_vbf_coords = labels[x.index(opt_vbf_point[0])]
    opt_ggf_coords = labels[y.index(opt_ggf_point[1])]

    print(
        "Optimal VBF Merit Point (xbb_cut_vbf, bdt_cut_vbf, xbb_cut_bin1, bdt_cut_bin1):",
        opt_vbf_coords,
        " at ",
        opt_vbf_point,
    )
    print(
        "Optimal ggF Merit Point (xbb_cut_vbf, bdt_cut_vbf, xbb_cut_bin1, bdt_cut_bin1):",
        opt_ggf_coords,
        " at ",
        opt_ggf_point,
    )

    plt.scatter(
        opt_vbf_point[0],
        opt_vbf_point[1],
        color="green",
        marker="s",
        s=100,
        label="Optimal VBF Merit",
    )
    plt.scatter(
        opt_ggf_point[0],
        opt_ggf_point[1],
        color="orange",
        marker="s",
        s=100,
        label="Optimal ggF Merit",
    )
    plt.text(opt_ggf_point[0], opt_ggf_point[1], f"opt ggf {opt_ggf_coords}", fontsize=9)
    plt.text(opt_vbf_point[0], opt_vbf_point[1], f"opt vbf {opt_vbf_coords}", fontsize=9)

    # Add axis labels and title
    plt.xlabel("Figure of Merit VBF")
    plt.ylabel("Figure of Merit ggF")
    plt.title("Scatter Plot with Pareto Front")
    plt.legend()

    plt.savefig(f"{path}/scatter_plot_with_pareto_front_bovers_real.png")

    print(len(results.keys()), len(pareto_front), len(points), len(x), len(y), len(labels))
    # print(results)


# for all points n paretor front and max, we will compute the limits for vbf and ggf and compare to see if we can get the same ggf and better vbf.,
# On a subrange of the points, we will compute the graph and also a histogram over each label so that we can refine the grid search.
def graph_with_cut(
    results,
    ggF_thresh=9.0,
    vbf_thresh=0.32,
    path="/home/users/annava/projects/HH4b/plots/PostProcess/24May16Joint",
):
    x = [
        2 * np.sqrt(value["nevents_bkg_vbf"]) / value["nevents_sig_vbf"]
        for value in results.values()
    ]
    y = [
        2 * np.sqrt(value["nevents_bkg_ggF"]) / value["nevents_sig_ggF"]
        for value in results.values()
    ]
    # x = [1/2*(value["figure_of_merit_vbf"]) for value in results.values()]
    # y = [1/2*(value["figure_of_merit_ggF"]) for value in results.values()]
    labels = list(results.keys())

    # Combine x and y into points
    points = list(zip(x, y))

    filtered_points = [(vbf, ggf) for vbf, ggf in points if ggf < ggF_thresh and vbf < vbf_thresh]
    filtered_labels = [
        labels[i] for i, (vbf, ggf) in enumerate(points) if ggf < ggF_thresh and vbf < vbf_thresh
    ]

    # print("Filtered points:", zip(filtered_labels,filtered_points))

    # Plotting the filtered points
    filtered_x = [point[0] for point in filtered_points]
    filtered_y = [point[1] for point in filtered_points]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="All Points from Scan")
    plt.scatter(
        filtered_x,
        filtered_y,
        color="red",
        edgecolor="black",
        marker="o",
        s=50,
        label="Filtered Points",
    )

    # Label each point with its coordinate for filtered points
    # for i in range(len(filtered_labels)):
    #     plt.text(filtered_x[i], filtered_y[i], f"{filtered_labels[i]}", fontsize=9)

    # Add axis labels and title
    plt.xlabel("Figure of Merit VBF ($2\sqrt{B}/S$)")
    plt.ylabel("Figure of Merit ggF ($2\sqrt{B}/S$)")
    plt.title(f"Scatter Plot with Filtered Points (cut: {ggF_thresh}, {vbf_thresh})")
    plt.legend()
    plt.savefig(f"{path}//scatter_plot_cut_{ggF_thresh}_{vbf_thresh}.png")

    print("len of labels: filtered_labels", len(filtered_labels), len(labels))
    xbb_cut_vbf = [float(label[0]) for label in filtered_labels]
    bdt_cut_vbf = [float(label[1]) for label in filtered_labels]
    xbb_cut_bin1 = [float(label[2]) for label in filtered_labels]
    bdt_cut_bin1 = [float(label[3]) for label in filtered_labels]

    print(f"Domain limits for xbb_cut_vbf: {min(xbb_cut_vbf)} to {max(xbb_cut_vbf)}")
    print(f"Domain limits for bdt_cut_vbf: {min(bdt_cut_vbf)} to {max(bdt_cut_vbf)}")
    print(f"Domain limits for xbb_cut_bin1: {min(xbb_cut_bin1)} to {max(xbb_cut_bin1)}")
    print(f"Domain limits for bdt_cut_bin1: {min(bdt_cut_bin1)} to {max(bdt_cut_bin1)}")

    # Calculate histogram data to find the maximum y-value\
    bin_num = 20
    hist_data = [
        np.histogram(xbb_cut_vbf, bins=bin_num),
        np.histogram(bdt_cut_vbf, bins=bin_num),
        np.histogram(xbb_cut_bin1, bins=bin_num),
        np.histogram(bdt_cut_bin1, bins=bin_num),
    ]

    max_y = max([35] + [max(hist[0]) for hist in hist_data])

    # Create histograms
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.hist(xbb_cut_vbf, bins=bin_num, color="blue", edgecolor="black")
    plt.title("Histogram of xbb_cut_vbf")
    plt.xlabel("xbb_cut_vbf")
    plt.ylabel("Frequency")
    plt.xlim(0.9, 1)
    plt.ylim(0, max_y)

    plt.subplot(2, 2, 2)
    plt.hist(bdt_cut_vbf, bins=bin_num, color="green", edgecolor="black")
    plt.title("Histogram of bdt_cut_vbf")
    plt.xlabel("bdt_cut_vbf")
    plt.ylabel("Frequency")
    plt.xlim(0.9, 1)
    plt.ylim(0, max_y)

    plt.subplot(2, 2, 3)
    plt.hist(xbb_cut_bin1, bins=bin_num, color="red", edgecolor="black")
    plt.title("Histogram of xbb_cut_bin1")
    plt.xlabel("xbb_cut_bin1")
    plt.ylabel("Frequency")
    plt.xlim(0.9, 1)
    plt.ylim(0, max_y)

    plt.subplot(2, 2, 4)
    plt.hist(bdt_cut_bin1, bins=bin_num, color="purple", edgecolor="black")
    plt.title("Histogram of bdt_cut_bin1")
    plt.xlabel("bdt_cut_bin1")
    plt.ylabel("Frequency")
    plt.xlim(0.9, 1)
    plt.ylim(0, max_y)
    # print(filtered_labels)

    plt.suptitle(
        f"Histograms with ggF_thresh={ggF_thresh} and vbf_thresh={vbf_thresh}", fontsize=16
    )
    plt.tight_layout()
    plt.savefig(f"{path}//histogram_cut_{ggF_thresh}_{vbf_thresh}.png")
    plt.close()


if __name__ == "__main__":
    file_path = (
        "/home/users/annava/projects/HH4b/plots/PostProcess/24May16Joint/results_vbf_bin1_10000.pkl"
    )
    file_path = (
        "/home/users/annava/projects/HH4b/plots/PostProcess/24May30Joint/results_vbf_bin1_50625.pkl"
    )
    file_path = "/home/users/annava/projects/HH4b/plots/PostProcess/24Jun06/results_vbf_bin1_updated_40000.pkl"
    # file_path = "/home/users/annava/projects/HH4b/plots/PostProcess/24Jun06/results_vbf_bin1_updated_40000_updated.pkl"
    with Path.open(file_path, "rb") as file:
        results = pickle.load(file)

    graph_with_pareto_front(
        results, path="/home/users/annava/projects/HH4b/plots/PostProcess/24Jun06"
    )
    # graph_with_cut(results,ggF_thresh = 100, vbf_thresh = 100,path="/home/users/annava/projects/HH4b/plots/PostProcess/24Jun06")
    # graph_with_cut(results,ggF_thresh = 10, vbf_thresh = 0.2,path="/home/users/annava/projects/HH4b/plots/PostProcess/24Jun06")
    # for ggF_thresh in [9.0, 9.1,10]:
    #     for vbf_thresh in [0.32, 0.35, 0.4]:
    #         graph_with_cut(results,ggF_thresh = ggF_thresh, vbf_thresh = vbf_thresh,path="/home/users/annava/projects/HH4b/plots/PostProcess/24May30Joint")

    # /home/users/annava/projects/HH4b/src/HH4b/postprocessing/results_vbf_bin1_updated_40000.pkl
