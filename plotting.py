import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle

from collections import defaultdict

def smooth_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 50))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def plot_results(domain_name, learning_name, all_results, outdir="results",
                 smooth=False, dist=False, llm_queries=None):
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
    if llm_queries is not None:
        llm_ys = []
        llm_xs = []
        for iter, num_accept in llm_queries:
            if num_accept > 0:
                llm_ys.append(results_mean[iter])
                llm_xs.append(iter)
        plt.scatter(llm_xs, llm_ys, c='#2ca02c')

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
    plt.close()
    print("Wrote out to {}".format(outfile))

from settings import PlottingConfig as pc
def main(results_path):
    """Plot the results in results/, specified by settings."""
    figures = []
    for domain, methods, seeds in zip(pc.domains, pc.methods, pc.seeds):
        lines = []
        for m,s in zip(methods, seeds):
            learning_name, curiosity_name = m
            lines.append(PlotLine(curiosity_name, learning_name, s))
        save_dir = f'plots/{domain}'
        os.makedirs(save_dir, exist_ok=True)
        figures.append(Figure(domain, lines, save_dir))
    missing_seeds = set()
    for figure in figures:
        ms = figure.run(results_path)
        missing_seeds |= ms
    print(f"Missing seeds:\n\t" + "\n\t".join(sorted(missing_seeds)))
    
import dataclasses

@dataclasses.dataclass
class PlotLine:
    def __init__(self, curiosity_method, learning_method, seeds):
        self.curiosity_method = curiosity_method
        self.learning_method = learning_method
        self.seeds = seeds
        
class Figure:
    def __init__(self, domain, plotlines:list[PlotLine], save_dir):
        self.domain = domain
        self.plotlines = plotlines
        self.save_dir = save_dir

    def run(self, results_path):
        domain = self.domain
        missing_seeds = set()
        outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), results_path)
        domain_path = os.path.join(results_path, domain)
        min_seeds = np.inf
        max_seeds = 0
        plt.figure()
        number_of_colors = len(self.plotlines)
        ax = plt.gca()
        colors = [next(ax._get_lines.prop_cycler)['color'] for _ in range(number_of_colors)]
        color_idx = 0
        for plotline in self.plotlines:
            learner = plotline.learning_method
            if learner == 'LLMWarmStart+LNDR':
                name = f'{domain}_seeds{plotline.seeds[0]}-{plotline.seeds[-1]}_{plotline.curiosity_method}_succ.png'
            explorer = plotline.curiosity_method
            seeds = plotline.seeds
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
                for seed in seeds:
                    missing_seeds.add(f"\t{domain}\t{learner}\t{explorer} Seed {seed}")
                return missing_seeds
            for i,r in enumerate(results):
                results[i] = r[:min_length]
            results = np.array(results)
            label = f"{learner}, {explorer}"
            xs = results[0,:,0]
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
        
        plt.ylabel("Success rate on test problems")
        plt.title(title)
        plt.ylim((-0.1, 1.1))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.xlabel("Iterations")

        outfile = os.path.join(self.save_dir, name)
        plt.savefig(outfile, dpi=300)
        print("Wrote out to {}".format(outfile))
        plt.close()
        return missing_seeds

if __name__ == "__main__":
    import argparse
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--individual_plots', action='store_true')
    parser.add_argument("-p", "--planning_results", action='store_true')
    args = parser.parse_args()

    if args.planning_results:
        path = 'results/planning_ops'
    else:
        path = 'results_openstack/results'
    llm_path = 'results/llm_iterative_log'

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

 
                # for i,f in enumerate(os.listdir(f"{path}/{domain_name}/{learning_name}/{curiosity_name}")):
                for seed in pc.seeds[0]:
                    all_results = defaultdict(list)
                    results_path = os.path.join(f"{path}/{domain_name}/{learning_name}/{curiosity_name}",f'{domain_name}_{learning_name}_{curiosity_name}_{seed}.pkl')
                    if not os.path.exists(results_path):
                        print(f"Missing seed {seed} for domain {domain_name} learner {learning_name} curiosity {curiosity_name}")
                        continue
                    with open(results_path, 'rb') as fh:
                        all_results[curiosity_name].append(pickle.load(fh))


                    llm_queries = None
                    if learning_name == 'LLM+LNDR' or learning_name == "LLMIterative+LNDR" or learning_name == "LLMIterative+ZPK":
                        p = os.path.join(llm_path, domain_name, curiosity_name, str(seed), 'experiment0', 'llm_ops_accepted.pkl') 
                        if os.path.exists(p):
                            with open(p, 'rb') as f:
                                llm_queries = pickle.load(f)
                    plot_results(f"{domain_name}{seed}", learning_name, all_results, outdir=succ_out, dist=False, llm_queries=llm_queries)
                    plot_results(f"{domain_name}{seed}", learning_name, all_results, outdir=dist_out, dist=True, llm_queries=llm_queries)