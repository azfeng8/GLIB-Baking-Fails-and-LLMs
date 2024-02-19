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
import random
def main(results_path):
    """Plot the results in results/, specified by settings."""
    missing_seeds = set()
    for domain in pc.domains:
        outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
        domain_path = os.path.join(results_path, domain)
        min_seeds = np.inf
        max_seeds = 0
        for dist_succ in ["dist", "succ"]:
            plt.figure()
            number_of_colors = len(pc.learner_explorer)
            ax = plt.gca()
            # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
            colors = [next(ax._get_lines.prop_cycler)['color'] for _ in range(number_of_colors)]
            color_idx = 0
            for seeds, learner_explorer in zip(pc.seeds, pc.learner_explorer):
                learner, explorer = learner_explorer
                seeds_path = os.path.join(domain_path, learner, explorer)
                results = []
                min_length = np.inf
                for seed in seeds:
                    pkl_fname = os.path.join(seeds_path, f'{domain}_{learner}_{explorer}_{str(seed)}.pkl')
                    if not os.path.exists(pkl_fname):
                        missing_seeds.add(f"\t{domain}\t{learner}\t{explorer} Seed {seed}")
                        continue
                    with open(pkl_fname, "rb") as f:
                        saved_results = pickle.load(f)
                        if len(saved_results) < min_length:
                            min_length = len(saved_results)
                    results.append(saved_results)
                min_seeds = min(min_seeds, len(results))
                max_seeds = max(max_seeds, len(results))
                if len(results) == 0:
                    raise Exception(f"Data not found: {seeds_path}")
                for i,r in enumerate(results):
                    results[i] = r[:min_length]
                results = np.array(results)
                label = f"{learner}, {explorer}"
                xs = results[0,:,0]
                if dist_succ == "dist":
                    ys = results[:, :, 2]
                else:
                    ys = results[:, :, 1]
                results_mean = np.mean(ys, axis=0)
                std = np.std(ys, axis=0)
                std_top = results_mean + std
                std_bot = results_mean - std
                plt.plot(xs, results_mean, label=label.replace("_", " "), color=colors[color_idx])
                plt.fill_between(xs, std_bot, std_top, alpha=0.3, color=colors[color_idx])
                color_idx += 1
            if min_seeds == max_seeds:
                title = f"{domain} Domain ({min_seeds} seeds)"
            else:
                title = f"{domain} Domain, ({min_seeds} to {max_seeds} seeds)"
            
            if dist_succ == 'dist':
                plt.ylabel("Variational distance to 'true' transition model") 
            if dist_succ == 'succ':
                plt.ylabel("Success rate on test problems")
            plt.title(title)
            plt.ylim((-0.1, 1.1))
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.xlabel("Iterations")
 
            outfile = os.path.join(outdir, "{}_{}.png".format(
                domain, dist_succ))
            plt.savefig(outfile, dpi=300)
            print("Wrote out to {}".format(outfile))
    print(f"Missing seeds:\n\t" + "\n\t".join(sorted(missing_seeds)))

    
if __name__ == "__main__":
    import argparse
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--individual_plots', action='store_true')
    parser.add_argument("-p", "--planning_results", action='store_true')
    args = parser.parse_args()

    if args.planning_results:
        path = 'results/planning_results'
    else:
        path = 'results/results/'

    if not args.individual_plots:
        main(path)
    else:
    ### Make individual plots
        for domain_name in pc.domains:
            for learning_name, curiosity_name in pc.learner_explorer:
                outdir = f"individual_plots/{domain_name}/{learning_name}/{curiosity_name}"
                succ_out = f"{outdir}/succ"
                dist_out = f"{outdir}/dist"
                if os.path.exists(succ_out):
                    shutil.rmtree(succ_out)
                if os.path.exists(dist_out):
                    shutil.rmtree(dist_out)
                os.makedirs(succ_out, exist_ok=True)
                os.makedirs(dist_out, exist_ok=True)
 
                for i,f in enumerate(os.listdir(f"{path}/{domain_name}/{learning_name}/{curiosity_name}")):
                    all_results = defaultdict(list)
                    num = f.rstrip(".pkl").split("_")[-1]
                    with open(os.path.join(f"{path}/{domain_name}/{learning_name}/{curiosity_name}",f), 'rb') as fh:
                        all_results[curiosity_name].append(pickle.load(fh))
                    plot_results(f"{domain_name}{num}", learning_name, all_results, outdir=succ_out, dist=False)
                    plot_results(f"{domain_name}{num}", learning_name, all_results, outdir=dist_out, dist=True)