import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle

from collections import defaultdict

from settings import PlottingConfig as pc

sns.set(style="darkgrid")

def smooth_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 50))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def plot_results(domain_name, learning_name, all_results, outdir="results",
                 smooth=False, dist=False):
    """Results are lists of single-run result lists, across different
    random seeds.
    """
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), outdir)
    outfile = os.path.join(outdir, "{}_{}_{}.png".format(
        domain_name, learning_name, "dist" if dist else "succ"))
    plt.figure()
    if dist:
        ylabel = "Test Set Average Variational Distance"
    else:
        ylabel = "Test Set Success Rate"
    plt.ylabel(ylabel)

    for curiosity_module in sorted(all_results):
        results = np.array(all_results[curiosity_module])
        if len(results) == 0:
            continue
        label = curiosity_module
        xs = results[0, :, 0]
        if dist:
            ys = results[:, :, 2]
        else:
            ys = results[:, :, 1]
        results_mean = np.mean(ys, axis=0)
        # results_std = np.std(ys, axis=0)
        if smooth:
            xs, results_mean = smooth_curve(xs, results_mean)
            # _, results_std = smooth_curve(xs, results_std)
        plt.plot(xs, results_mean, label=label.replace("_", " "))
        # plt.fill_between(xs, results_mean+results_std,
        #                  results_mean-results_std, alpha=0.2)
    min_seeds = min(len(x) for x in all_results.values())
    max_seeds = max(len(x) for x in all_results.values())
    if min_seeds == max_seeds:
        title = "{} Domain, {} Learner ({} seeds)".format(
            domain_name, learning_name, min_seeds)
    else:
        title = "{} Domain, {} Learner ({} to {} seeds)".format(
            domain_name, learning_name, min_seeds, max_seeds)
    if smooth:
        title += " [smoothed]"
    plt.title(title)

    plt.ylim((-0.1, 1.1))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print("Wrote out to {}".format(outfile))

def main():
    """Plot the results in results/, specified by settings."""
    results_path = "results"
    min_seeds = np.inf
    max_seeds = 0
    for domain in pc.domains:
        outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
        domain_path = os.path.join(results_path, domain)
        for dist_succ in ["dist", "succ"]:
            plt.figure()
            for learner, explorer in pc.learner_explorer:
                seeds_path = os.path.join(domain_path, learner, explorer)
                results = []
                for pkl_fname in glob.glob(os.path.join(seeds_path, "*.pkl")):
                    with open(pkl_fname, "rb") as f:
                        saved_results = pickle.load(f)
                    results.append(saved_results)
                min_seeds = min(min_seeds, len(results))
                max_seeds = max(max_seeds, len(results))
                results = np.array(results)
                label = f"{learner}, {explorer}"
                xs = results[0,:,0]
                if dist_succ == "dist":
                    ys = results[:, :, 2]
                else:
                    ys = results[:, :, 1]
                results_mean = np.mean(ys, axis=0)
                plt.plot(xs, results_mean, label=label.replace("_", " "))
                if min_seeds == max_seeds:
                    title = f"{domain} Domain ({min_seeds} seeds)"
                else:
                    title = f"{domain} Domain, ({min_seeds} to {max_seeds} seeds)"
                plt.title(title)
                plt.ylim((-0.1, 1.1))
                plt.legend(loc="lower right")
                plt.tight_layout()
            outfile = os.path.join(outdir, "{}_{}.png".format(
                domain, dist_succ))
            plt.savefig(outfile, dpi=300)
            print("Wrote out to {}".format(outfile))

    
if __name__ == "__main__":
    main()