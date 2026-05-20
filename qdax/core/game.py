# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import

# %%
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm
import multiprocessing as mp
import queue

# from qdax.core.plot import *
import qdax.core.utils as utils
from copy import deepcopy
from functools import partial

# from scipy.spatial import cKDTree
from scipy.spatial.ckdtree import cKDTree
import threading
import time

# for pareto Front, use NSGA-III
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival


# %% [markdown]
# # GAME


# %%
def _compute_distance(distance_function, b, centroids):
    distances = distance_function(b, centroids)
    c_id = np.argmin(distances)
    return distances[c_id], c_id


# %% [markdown]
# ## Classic Archive


# %%
def cvt(k, dim, coef=10, verbose=False, rep=0):
    root = "../../data/cvt/"
    name = f"{int(k)}_{int(dim)}_{rep}.pk"

    if os.path.exists(root + name):
        with open(root + name, "rb") as f:
            X = pickle.load(f)
    else:
        rng = np.random.default_rng(rep)
        x = rng.random((k * coef, dim))
        k_means = KMeans(
            init="k-means++", n_clusters=k, n_init=1, verbose=False
        )  # ,algorithm="full")
        k_means.fit(x)
        X = k_means.cluster_centers_
        with open(root + name, "wb") as f:
            pickle.dump(X, f)
    return X


# %%
class A:  # Classic CVT Archive
    def __init__(self, config, rng, seed):
        self.rng = rng
        self.seed = seed
        self.n_cells = config["n_cells"]
        self.n_solution_dim = config["n_solution_dim"]
        self.n_behavior_dim = config["n_behavior_dim"]
        self.compute_distance = partial(_compute_distance, config["distance_function"])
        self.compare_fitness = config["compare_fitness"]
        self.cells_fitness = [None for _ in range(self.n_cells)]
        self.cells_solution = (
            {}
            if self.n_solution_dim is None
            else np.empty(shape=(self.n_cells, self.n_solution_dim))
        )
        self.cells_behavior = np.empty(shape=(self.n_cells, self.n_behavior_dim))
        self.cells_log_id = np.ones(shape=(self.n_cells), dtype=np.int32) * -1
        self.non_empty_cells = []
        self.centroids = cvt(self.n_cells, self.n_behavior_dim, rep=self.seed)

    def n_elites(self):
        return len(self.non_empty_cells)

    def sample_parent(self):
        p = self.rng.choice(self.non_empty_cells)
        return self.cells_solution[p], self.cells_log_id[p]

    def set_new_elite(self, cell_id, evaluation, append=True, reset=True):
        self.cells_log_id[cell_id] = evaluation["id"]
        self.cells_fitness[cell_id] = evaluation["fitness"]
        self.cells_solution[cell_id] = evaluation["solution"]
        self.cells_behavior[cell_id] = evaluation["behavior"]
        evaluation["is_elite"] = True

    def add_evaluation(self, evaluation):
        d, cell_id = self.compute_distance(evaluation["behavior"], self.centroids)
        if self.compare_fitness(evaluation["fitness"], self.cells_fitness[cell_id]):
            self.set_new_elite(cell_id, evaluation)
            if cell_id not in self.non_empty_cells:  # keeps track of the filled cells
                self.non_empty_cells.append(cell_id)


# %% [markdown]
# ## Dominated Novelty Search


# %%
class DNS:  # Classic CVT Archive
    def __init__(self, config, rng, seed):
        self.rng = rng
        self.seed = seed
        self.n_solution_dim = config["n_solution_dim"]
        self.n_behavior_dim = config["n_behavior_dim"]
        self.n_cells = config["n_cells"]
        self.k = config["k"]
        self.compute_distances = config["distance_function"]
        self.cells_fitness = [None for _ in range(self.n_cells)]
        self.cells_solution = (
            {}
            if self.n_solution_dim is None
            else np.empty(shape=(self.n_cells, self.n_solution_dim))
        )
        self.cells_behavior = np.empty(shape=(self.n_cells, self.n_behavior_dim))
        self.cells_log_id = np.ones(shape=(self.n_cells), dtype=np.int32) * -1
        self.non_empty_cells = []

    def n_elites(self):
        return len(self.non_empty_cells)

    def sample_parent(self):
        p = self.rng.choice(self.non_empty_cells)
        return self.cells_solution[p], self.cells_log_id[p]

    def set_new_elite(self, cell_id, evaluation):
        self.cells_log_id[cell_id] = evaluation["id"]
        self.cells_fitness[cell_id] = evaluation["fitness"]
        self.cells_solution[cell_id] = evaluation["solution"]
        self.cells_behavior[cell_id] = evaluation["behavior"]
        evaluation["is_elite"] = True

    def compute_dominated_fitness(self, fitnesses, behaviors):
        dominated_f = np.empty(len(fitnesses))
        for i in range(len(fitnesses)):
            indices_fitter = np.where(fitnesses > fitnesses[i])[0]
            if len(indices_fitter) == 0:
                dominated_f[i] = np.inf
            else:
                d = self.compute_distances(behaviors[i], behaviors[indices_fitter])
                if self.k < len(indices_fitter):
                    indices_nn = np.argpartition(
                        d, self.k - 1
                    )[
                        : self.k
                    ]  # argpartition guarantees that the kth element is in sorted position and all smaller elements will be moved before it. Thus the first k elements will be the k-smallest elements.
                    dominated_f[i] = np.mean(d[indices_nn])
                else:
                    dominated_f[i] = np.mean(d)
        return dominated_f

    def add_evaluation(self, evaluation):
        if (
            len(self.non_empty_cells) < self.n_cells
        ):  # initialisation of the n_cells with the first n different behaviors
            cell_id = len(self.non_empty_cells)
            self.set_new_elite(cell_id, evaluation)
            self.non_empty_cells.append(cell_id)
        else:  # we need to remove the worst individual
            fitnesses = np.concatenate([self.cells_fitness, [evaluation["fitness"]]])  # type: ignore
            behaviors = np.concatenate([self.cells_behavior, [evaluation["behavior"]]])
            dominated_f = self.compute_dominated_fitness(fitnesses, behaviors)
            idx_worst = np.argmin(dominated_f)
            if idx_worst != self.n_cells:  # the new evaluation is not the worst
                self.set_new_elite(idx_worst, evaluation)


# %% [markdown]
# ## Growing Archive


# %%
def compute_distances(distance_function, centroids):
    n = len(centroids)
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = distance_function(centroids[i], centroids)
    return distances


def _compute_neighbors(distance_function, centroids):
    distances = compute_distances(distance_function, centroids)
    c_ids = np.argsort(distances, axis=1)
    distances = np.array([distances[i][c_ids[i]] for i in range(len(centroids))])
    return distances, c_ids


# %%
class GA:
    def __init__(self, config, rng, seed):
        self.rng = rng
        self.seed = seed

        self.n_cells = config["n_cells"]
        self.n_solution_dim = config["n_solution_dim"]
        self.n_behavior_dim = config["n_behavior_dim"]
        self.use_redristribution = config["use_redristribution"]
        self.use_collection = config["use_collection"]
        self.use_repair = config["use_repair"]
        self.compute_distance = partial(_compute_distance, config["distance_function"])
        self.compute_neighbors = partial(
            _compute_neighbors, config["distance_function"]
        )
        self.compare_fitness = config["compare_fitness"]
        self.cells_fitness = [None for _ in range(self.n_cells)]
        self.cells_solution = (
            {}
            if self.n_solution_dim is None
            else np.empty(shape=(self.n_cells, self.n_solution_dim))
        )
        self.cells_behavior = np.empty(shape=(self.n_cells, self.n_behavior_dim))
        self.cells_log_id = np.ones(shape=(self.n_cells), dtype=np.int32) * -1
        self.cells_former_elites = {i: [] for i in range(self.n_cells)}
        self.non_empty_cells = []
        self.centroids = np.empty(shape=(self.n_cells, self.n_behavior_dim))
        self.n_centroids = 0
        self.dmin = None

    def n_elites(self):
        return len(self.non_empty_cells)

    def sample_parent(self):
        p = self.rng.choice(self.non_empty_cells)
        return self.cells_solution[p], self.cells_log_id[p]

    def fill_empty_cell(self, evaluation):
        cell_id = np.where(
            np.linalg.norm(
                self.centroids[: self.n_centroids] - evaluation["behavior"], axis=1
            )
            == 0
        )[0]
        if len(cell_id) == 0:  # if not already present creates a new cell
            self.centroids[self.n_centroids] = evaluation["behavior"]
            self.cells_former_elites[self.n_centroids] = [evaluation]
            cell_id = self.n_centroids
            self.n_centroids += 1
            self.non_empty_cells.append(cell_id)
        else:
            cell_id = cell_id[0]
        return cell_id

    def set_new_elite(self, cell_id, evaluation, append=True, reset=True):
        self.cells_log_id[cell_id] = evaluation["id"]
        self.cells_fitness[cell_id] = evaluation["fitness"]
        self.cells_solution[cell_id] = evaluation["solution"]
        self.cells_behavior[cell_id] = evaluation["behavior"]
        evaluation["is_elite"] = True
        if append:
            self.cells_former_elites[cell_id].append(deepcopy(evaluation))
        elif reset:
            self.cells_former_elites[cell_id] = [deepcopy(evaluation)]

    def compute_dmin(self):
        self.d_neighbors, self.c_id_neighbors = self.compute_neighbors(self.centroids)
        self.dmin = np.min(self.d_neighbors[:, 1])

    def apply_collection(self, pruned):
        """collect elites from neighbors"""
        for j in range(self.n_cells):
            if len(self.cells_former_elites[j]) > 1 and j != pruned:
                new_cell_elites, new_elite = [], None
                for elite in self.cells_former_elites[j]:
                    _, elite_id = self.compute_distance(
                        elite["behavior"], self.centroids
                    )
                    if (
                        elite_id == pruned
                    ):  # split the elites between the old and new cells
                        self.cells_former_elites[pruned].append(elite)
                        if self.compare_fitness(
                            elite["fitness"], self.cells_fitness[pruned]
                        ):  # bootstrap the new cell with already found elites from neighbors
                            self.set_new_elite(pruned, elite, reset=False, append=False)
                    elif (
                        j == elite_id
                    ):  # repair the archive at the new cell location (neighbor cells can loose their elites)
                        new_cell_elites.append(elite)
                        if new_elite is None or self.compare_fitness(
                            elite["fitness"], new_elite["fitness"]
                        ):
                            new_elite = elite
                    else:
                        print("Should not go here!")
                self.set_new_elite(j, new_elite, reset=False, append=False)
                self.cells_former_elites[j] = new_cell_elites

    def apply_repair(self, pruned):
        """soft sub-optimal repair in case the current elite of cell is stolen by the new one"""
        for j in range(self.n_cells):
            if len(self.cells_former_elites[j]) > 1 and j != pruned:
                _, elite_id = self.compute_distance(
                    self.cells_former_elites[j][-1]["behavior"], self.centroids
                )
                if (
                    elite_id == pruned
                ):  # if the new cell steals the current elite we reinstate the initial centroid
                    self.set_new_elite(
                        j, self.cells_former_elites[j][0], reset=True, append=False
                    )

    def apply_redristribution(self, former_elites):
        """distribute the pruned elites into their new cells"""
        for elite in former_elites:
            _, cell_id = self.compute_distance(elite["behavior"], self.centroids)
            if self.compare_fitness(elite["fitness"], self.cells_fitness[cell_id]):
                self.set_new_elite(cell_id, elite)

    def add_evaluation(self, evaluation):
        changed = False

        if (
            self.n_centroids < self.n_cells
        ):  # initialisation of the n_cells with the first n different behaviors
            cell_id = self.fill_empty_cell(evaluation)
            if (
                self.n_centroids == self.n_cells
            ):  # the archive is full, we can compute the minimal distance
                self.compute_dmin()
        else:  # only grows if the new solution is farther than the closest two cells
            d, cell_id = self.compute_distance(evaluation["behavior"], self.centroids)
            if d > self.dmin:
                centroid_A = np.argmin(self.d_neighbors[:, 1])
                centroid_B = self.c_id_neighbors[centroid_A, 1]
                pruned = (
                    centroid_A
                    if self.d_neighbors[centroid_A, 2] < self.d_neighbors[centroid_B, 2]
                    else centroid_B
                )
                self.centroids[pruned] = evaluation["behavior"]
                self.compute_dmin()
                former_elites = deepcopy(self.cells_former_elites[pruned])  # type: ignore
                changed = True
                self.set_new_elite(pruned, evaluation, reset=True, append=False)
                if self.use_redristribution:
                    self.apply_redristribution(former_elites)
                if not self.use_collection and self.use_repair:
                    self.apply_repair(pruned)
                if self.use_collection:
                    self.apply_collection(pruned)

        if (
            self.compare_fitness(evaluation["fitness"], self.cells_fitness[cell_id])
            and not changed
        ):  # classic elitess update if better fitness
            self.set_new_elite(cell_id, evaluation)


# %% [markdown]
# ## Multi-Task Archive


# %%
class MTA:
    def __init__(self, config, n_tasks, rng, seed):
        self.rng = rng
        self.n_tasks = n_tasks
        if config["archive_type"] == "GA":  # Growing archive
            self.archives = [GA(config, self.rng, seed) for _ in range(self.n_tasks)]
        elif config["archive_type"] == "DNS":  # Dominated Novelty Search
            self.archives = [DNS(config, self.rng, seed) for _ in range(self.n_tasks)]
        else:  # CVT MAP-Elites
            self.archives = [A(config, self.rng, seed) for _ in range(self.n_tasks)]
        self.non_empty_archive = []

    def update(self, evaluations):
        for evaluation in evaluations:
            if evaluation is not None:
                self.archives[evaluation["task_id"]].add_evaluation(evaluation)
                if evaluation["task_id"] not in self.non_empty_archive:
                    self.non_empty_archive.append(evaluation["task_id"])

    def sample_parents(self):
        p1_a_id, p2_a_id = self.rng.choice(self.non_empty_archive, 2)
        p1, p1_id = self.archives[p1_a_id].sample_parent()
        p2, p2_id = self.archives[p2_a_id].sample_parent()
        return p1, p2, (p1_a_id, p1_id), (p2_a_id, p2_id)

    def n_elites(self):
        return np.sum([archive.n_elites() for archive in self.archives])


# %%
def parallel_worker(evaluation_function, job_queue, res_queue):
    worker_id = mp.current_process()._identity[0]
    while True:
        args = job_queue.get()
        res = evaluation_function(
            tasks=args["tasks"], candidates=args["candidates"], worker_id=worker_id
        )
        res_queue.put(res)


# %%
class MTMB_ME:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config["seed"])
        self.tasks = config["tasks"]
        self.n_tasks = len(self.tasks)
        self.archive = MTA(
            config["archive_config"], self.n_tasks, self.rng, config["seed"]
        )
        self.evaluation_function = partial(
            config["env_config"]["evaluation_function"],
            **config["env_config"]["evaluation_config"],
        )
        self.sample_random = config["env_config"]["sample_random_function"]
        self.sample_crossover_and_mutation = config["env_config"][
            "crossover_and_mutation_function"
        ]
        self.n_proc = config["n_proc"]
        self.budget = config["budget"]
        self.batch_size = config[
            "batch_size"
        ]  # number of individuals evaluated in // on the same task
        self.parallel = config["parallel"]
        self.parallel_timeout = config["parallel_timeout"]
        self.verbose = config["verbose"]
        self.save_folder = config["save_folder"]
        self.log_interval = config["log_interval"]
        self.use_logging = config["use_logging"]
        self.init_elites = config["init_elites"]
        self.is_random = True
        self.it_end_random = ""
        self.it = 0
        self.log = []

    def save_archive(self, name=None):
        archive_save = {"fitness": [], "solutions": [], "behavior": [], "log_id": []}
        for archive in self.archive.archives:
            archive_save["fitness"].append(archive.cells_fitness)
            archive_save["solutions"].append(archive.cells_solution)
            archive_save["behavior"].append(archive.cells_behavior)
            archive_save["log_id"].append(archive.cells_log_id)
        archive_save["log"] = self.log
        archive_save["batch_size"] = self.batch_size
        archive_save["tasks"] = self.tasks
        save_name = self.save_folder + (
            "/archive_save.pk" if name is None else f"/archive_save_{name}.pk"
        )
        utils.save_pickle(save_name, archive_save)

    def sample_candidate(self):
        if self.is_random:  # initialization with random solutions till we find enough elites for diversity
            candidate, origin = self.sample_random(
                self.rng, self.config["env_config"]["random_sampling_config"]
            )
        else:
            p1, p2, p1_id, p2_id = self.archive.sample_parents()
            candidate, origin = self.sample_crossover_and_mutation(
                self.rng,
                p1,
                p2,
                self.config["env_config"]["crossover_and_mutation_config"],
            )
            origin["p1"] = p1_id
            origin["p2"] = p2_id
        return {"value": candidate, "origin": origin}

    def sample_task(self):
        task_id = self.rng.integers(self.n_tasks)
        return {"value": self.tasks[task_id], "id": task_id}

    def sample_tasks(self):
        indexes = self.rng.choice(
            np.arange(self.n_tasks),
            self.batch_size,
            replace=self.batch_size > self.n_tasks,
        )
        return [{"value": self.tasks[task_id], "id": task_id} for task_id in indexes]

    def sample_new_evaluations(self):
        tasks = self.sample_tasks()
        candidates = [self.sample_candidate() for _ in range(self.batch_size)]
        return tasks, candidates

    def update_archive(self, evaluations):
        has_changed = False
        for ev in evaluations:
            if ev is not None:
                ev["id"] = self.it
                self.it += 1
        self.archive.update(evaluations)
        if self.use_logging:
            self.logger(evaluations)
        if self.is_random and self.archive.n_elites() >= self.init_elites:
            self.is_random = False
            self.it_end_random = self.it
            has_changed = True
        return has_changed

    def logger(self, evaluations):
        self.log.append(evaluations)

    def get_fitness(self):
        return [ev["fitness"] for batch in self.log for ev in batch]

    def get_behaviors(self):
        return [ev["behavior"] for batch in self.log for ev in batch]

    def get_solutions(self):
        return [ev["solution"] for batch in self.log for ev in batch]

    def get_descriptions(self):
        return [ev["description"] for batch in self.log for ev in batch]

    def run(self):
        if self.parallel:
            job_queue = mp.Queue()
            res_queue = mp.Queue()
            pool = mp.Pool(
                self.n_proc,
                parallel_worker,
                (self.evaluation_function, job_queue, res_queue),
            )
            for _ in range(self.n_proc):
                tasks, candidates = self.sample_new_evaluations()
                job_queue.put({"tasks": tasks, "candidates": candidates})
        else:
            job_queue = [self.sample_new_evaluations()]

        if self.verbose > 0:  # create loading bar info
            loading_bar = tqdm(
                total=self.budget - self.it, ncols=100, smoothing=0.01, mininterval=1
            )
            loading_bar.set_description(
                "Random" if self.is_random else f"C&M [{self.it_end_random}]"
            )

        self.save_archive("init")
        last_it = self.it
        while self.it < self.budget:
            # collect the evaluation
            if self.parallel:
                try:
                    evaluated_candidates = res_queue.get(timeout=self.parallel_timeout)
                except queue.Empty:
                    print("Timeout: workers may be stuck. Restarting.")
                    pool.terminate()
                    pool.join()
                    # Optionally clear the job_queue and refill
                    job_queue = mp.Queue()
                    res_queue = mp.Queue()
                    pool = mp.Pool(
                        self.n_proc,
                        parallel_worker,
                        (self.evaluation_function, job_queue, res_queue),
                    )
                    for _ in range(self.n_proc):
                        tasks, candidates = self.sample_new_evaluations()
                        job_queue.put({"tasks": tasks, "candidates": candidates})
            else:
                tasks, candidates = job_queue.pop(0)
                evaluated_candidates = self.evaluation_function(
                    tasks=tasks, candidates=candidates
                )

            # update archive
            has_changed = self.update_archive(
                evaluated_candidates
            )  # has_changed = switch from random to Crossover and Mutation

            # log archive
            if self.it % self.log_interval == 0:
                self.save_archive(self.it)

            if self.verbose:  # update loading bar info
                loading_bar.update(self.it - last_it)
                last_it = self.it
                if (
                    has_changed
                ):  # Swhitch to Crossover and Mutation once the initialisation is done
                    loading_bar.set_description(f"C&M [{self.it}]")

            # sample new candidates batch
            tasks, candidates = self.sample_new_evaluations()
            if self.parallel:  # put a new job
                job_queue.put({"tasks": tasks, "candidates": candidates})
            else:
                job_queue.append((tasks, candidates))

        if self.verbose:
            loading_bar.close()

        if self.parallel:
            job_queue.close()
            res_queue.close()
            pool.terminate()
            pool.join()

        self.save_archive()


# %% [markdown]
# ## get_MTMB_config


# %%
def get_MTMB_ME_config(
    side,
    get_env_config,
    budget,
    n_cells,
    batch_size,
    init_elites,
    parallel,
    n_proc,
    parallel_timeout,
    archive_type="GA",
    seed_id=0,
    verbose=True,
    use_logging=False,
    log_interval=10_000,
    use_redristribution=False,
    use_collection=False,
    DNS_k=5,
):
    """
    get_env_config should countain:
        "sample_random_function": function to sample random solution (rng, config) -> solution (should be a dict), origin (where origin is a description used for analysis)
        "random_sampling_config": its dict of parameters,

        "crossover_and_mutation_function": variation operator (rng, sol1, sol2, config) -> solution, origin (where origin is a description used for analysis, for example both parents: WARNING! Make sure that the new solution is a new object and not a reference)  )
        "crossover_and_mutation_config":  its dict of parameters,

        "evaluation_function": evaluation function (**config, tasks, candidates, worker_id) -> evaluations ( where evaluation is a list of {"task_id", "id", "solution", "origin", "fitness", "behavior", "other_fitness", "other_behavior"})
        "evaluation_config":  its dict of parameters,
    """
    assert side in ["red", "blue"]
    seed = utils.seeds[seed_id]
    env_config = get_env_config(side, seed)
    config = {
        "budget": budget,
        "batch_size": batch_size,
        "init_elites": init_elites,
        "seed": seed,
        "verbose": verbose,
        "log_interval": log_interval,
        "use_logging": use_logging,
        "parallel": parallel,
        "parallel_timeout": parallel_timeout,
        "n_proc": n_proc,
        "archive_config": {
            "archive_type": archive_type,
            "n_cells": n_cells,
            "n_behavior_dim": env_config["behavior_dim"],
            "n_solution_dim": None,
            "use_redristribution": use_redristribution,
            "use_collection": use_collection,
            "k": DNS_k,
            "use_repair": True,
            "distance_function": env_config["distance_function"],
            "compare_fitness": env_config["compare_fitness"],
        },
        "env_config": env_config,
    }
    return config


# %% [markdown]
# ## Tournament

# %%
eval_keys = ["fitness", "behavior", "other_fitness", "other_behavior"]


# %% [markdown]
# ### With batch (like Jax vmap but could also be just one)


# %%
def compute_tournament_in_batch(config, reds, blues, verbose=True, log_level=1):
    tournament = {}
    for i, red in enumerate(reds):
        for j, blue in enumerate(blues):
            blue["generation"] = "red"
            tournament[(i, j)] = {
                "candidate": {"value": red, "origin": i},
                "task": {"id": j, "value": blue},
            }
    batch_size = config["batch_size"]
    all_keys = list(tournament.keys())
    t = (
        tqdm(range(0, len(tournament), batch_size))
        if verbose
        else range(0, len(tournament), batch_size)
    )
    for i in t:
        keys = all_keys[i : i + batch_size]
        if len(keys) < batch_size:
            keys += [keys[-1]] * (batch_size - len(keys))
        candidates = [tournament[key]["candidate"] for key in keys]
        tasks = [tournament[key]["task"] for key in keys]
        evaluations = config["env_config"]["evaluation_function"](
            **config["env_config"]["evaluation_config"],
            tasks=tasks,
            candidates=candidates,
        )
        for j, key in enumerate(keys):
            if log_level == 1:
                tournament[key]["eval"] = {
                    key: evaluations[j][key]
                    for key in eval_keys
                    if key in evaluations[j]
                }
            elif log_level == 0:
                tournament[key]["eval"] = {"fitness": evaluations[j]["fitness"]}
            else:
                tournament[key]["eval"] = evaluations[j]
    return tournament


# %% [markdown]
# ### With Multi-processing


# %%
def parallel_worker_with_key(evaluation_function, job_queue, res_queue):
    worker_id = mp.current_process()._identity[0]
    while True:
        try:
            job = job_queue.get(timeout=30)  # Timeout to check for shutdown
            if job is None:
                break
            key, args = job
            try:
                res = evaluation_function(
                    tasks=args["tasks"],
                    candidates=args["candidates"],
                    worker_id=worker_id,
                )
                res_queue.put((key, res), timeout=10)
            except Exception as e:
                print(f"Worker error processing {key}: {e}")
                res_queue.put((key, [None]), timeout=10)
        except queue.Empty:
            # Periodic check for shutdown - could check a shared flag here
            continue
        except Exception as e:
            print(f"Worker exception: {e}")
            break


# %%
def pool_join_with_timeout(pool, timeout=30):
    """Join pool with timeout using threading"""

    def join_target():
        pool.join()

    join_thread = threading.Thread(target=join_target)
    join_thread.daemon = True
    join_thread.start()
    join_thread.join(timeout)

    return not join_thread.is_alive()


def compute_tournament_multi_proc(
    config, reds, blues, max_concurrent=None, full_log=False
):
    """Process tournament continuously with better timeout handling"""
    n_proc = min(config["n_proc"], len(reds) * len(blues))

    if max_concurrent is None:
        max_concurrent = n_proc

    job_queue = mp.Queue(maxsize=max_concurrent)
    res_queue = mp.Queue()

    pool = mp.Pool(
        n_proc,
        parallel_worker_with_key,
        (
            partial(
                config["env_config"]["evaluation_function"],
                **config["env_config"]["evaluation_config"],
            ),
            job_queue,
            res_queue,
        ),
    )

    tournament = {}
    all_pairs = []
    collected_pairs = {}
    # Generate all pairs
    for i, red_team in enumerate(reds):
        for j, blue_team in enumerate(blues):
            blue_team["generation"] = "red"
            task = {"id": j, "value": blue_team}
            candidate = {"value": red_team, "origin": i}
            all_pairs.append((i, j))
            collected_pairs[(i, j)] = False
            tournament[(i, j)] = {"candidate": candidate, "task": task}

    total_pairs = len(all_pairs)
    jobs_submitted = 0
    results_collected = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 3

    with tqdm(total=total_pairs, desc="Tournament") as pbar:
        # Submit initial batch
        while jobs_submitted < min(max_concurrent, total_pairs):
            key = all_pairs[jobs_submitted]
            job_queue.put(
                (
                    key,
                    {
                        "tasks": [tournament[key]["task"]],
                        "candidates": [tournament[key]["candidate"]],
                    },
                )
            )
            jobs_submitted += 1

        jobs_submitted = jobs_submitted % total_pairs

        # Continuous processing
        while (
            results_collected < total_pairs
            and consecutive_timeouts < max_consecutive_timeouts
        ):
            try:
                # Use shorter timeout but don't give up immediately
                key, evaluations = res_queue.get(timeout=30)
                ev = evaluations[0]
                consecutive_timeouts = 0  # Reset timeout counter

                if (
                    key in tournament and not collected_pairs[key]
                ):  # make sure not to collect the same pair twice
                    if not full_log:
                        tournament[key]["eval"] = {
                            key: ev[key] for key in eval_keys if key in evaluations[j]
                        }
                    else:
                        tournament[key]["eval"] = ev
                    results_collected += 1
                    pbar.update(1)
                    collected_pairs[key] = True

                # Submit next job if available
                if results_collected < total_pairs:
                    try:
                        key = all_pairs[jobs_submitted]
                        job_queue.put(
                            (
                                key,
                                {
                                    "tasks": [tournament[key]["task"]],
                                    "candidates": [tournament[key]["candidate"]],
                                },
                            ),
                            timeout=0.1,
                        )
                        jobs_submitted = (jobs_submitted + 1) % total_pairs
                        while collected_pairs[
                            all_pairs[jobs_submitted]
                        ]:  # there shouldn't be any infinite loop because np.sum(collected_pairs) == results_collected < total_pairs
                            jobs_submitted = (jobs_submitted + 1) % total_pairs

                    except queue.Full:
                        pass  # Will try again next iteration

            except queue.Empty:
                consecutive_timeouts += 1
                print(
                    f"Timeout #{consecutive_timeouts}. Progress: {results_collected}/{total_pairs}"
                )

                # If we've submitted all jobs but haven't collected all results,
                # and we're getting timeouts, some jobs might be stuck
                if jobs_submitted == total_pairs:
                    print("All jobs submitted, waiting for remaining results...")
                    # Give more time for remaining jobs
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        print("Too many consecutive timeouts, stopping collection")
                        break

    # Clean up failed evaluations
    to_del = [key for key, val in tournament.items() if val["eval"] is None]

    for key in to_del:
        del tournament[key]

    # empty job queue
    drained_count = 0  # drained count wasn't defined
    try:
        while True:
            try:
                job_queue.get_nowait()  # Non-blocking get
                drained_count += 1
            except queue.Empty:
                break
    except Exception as e:
        print(f"Error draining queue: {e}")

    # Send sentinel values to stop workers
    for _ in range(n_proc):
        try:
            job_queue.put(None, timeout=2)
        except queue.Full:
            pass

    # Quick fix - add timeouts to cleanup
    job_queue.close()
    res_queue.close()
    pool.terminate()

    if not pool_join_with_timeout(pool, timeout=30):
        print("Pool didn't terminate cleanly, killing processes")
        for process in pool._pool:  # ty:ignore[unresolved-attribute]
            if process.is_alive():
                process.kill()

        # Give kills time to take effect, then final join
        time.sleep(1)
        try:
            pool.join()  # Should be quick now
        except Exception:
            pass
    return tournament


# %% [markdown]
# ## GAME

# %% [markdown]
# ### Next generation selection

# %% [markdown]
# #### From touranement score or ranking


# %%
def select_new_tasks(score, n_tasks, F, criterion):
    assert criterion in ["quality", "diversity"]
    kmeans = KMeans(n_clusters=n_tasks).fit(score)
    centroids = kmeans.cluster_centers_
    tree = cKDTree(centroids)
    new_tasks = [None for _ in range(n_tasks)]
    current_value = [np.inf for _ in range(n_tasks)]
    for elite_id in range(len(score)):
        d, c_id = tree.query(score[elite_id], k=1)
        if criterion == "quality":  # pick the elite of the cell
            if new_tasks[c_id] is None or np.mean(F[elite_id]) > np.mean(
                F[new_tasks[c_id]]
            ):
                new_tasks[c_id] = elite_id
        else:  # pick the closest to the centroid
            if new_tasks[c_id] is None or d < current_value[c_id]:
                new_tasks[c_id] = elite_id
                current_value[c_id] = d
    return [x for x in new_tasks if x is not None]


# %% [markdown]
# #### From Pareto Front


# %%
def select_solutions_nsga3(objectives, k):
    """
    Select k solutions from n solutions using NSGA-III approach.

    This handles both cases:
    - If Pareto front < k: uses ranking + reference directions
    - If Pareto front >= k: selects diverse subset from front

    Parameters:
    -----------
    objectives : np.ndarray of shape (n, m)
        Objective values for n solutions on m objectives (to be minimized)
    k : int
        Number of solutions to select

    Returns:
    --------
    selected_indices : np.ndarray
        Indices of selected solutions
    """
    n, m = objectives.shape

    if k >= n:
        return np.arange(n)

    # Create reference directions
    # n_partitions controls the granularity (more = more reference directions)
    # Adjust based on k and m
    ref_dirs = get_reference_directions(m, k)

    # Create a dummy problem to satisfy pymoo's requirements
    from pymoo.core.problem import Problem

    class DummyProblem(Problem):
        def __init__(self, n_obj):
            super().__init__(n_var=1, n_obj=n_obj, n_constr=0, xl=0, xu=1)

        def _evaluate(self, x, out, *args, **kwargs):
            pass

    problem = DummyProblem(m)

    # Create a Population object (pymoo's data structure)
    # We need to create dummy individuals with the objectives
    from pymoo.core.individual import Individual

    pop = Population()
    for i in range(n):
        ind = Individual()
        ind.F = objectives[i]  # F is the objective values
        pop = Population.merge(pop, Population.create(ind))

    # Use NSGA-III survival mechanism
    survival = ReferenceDirectionSurvival(ref_dirs)

    # Select k solutions
    selected_pop = survival.do(
        problem=problem, pop=pop, n_survive=k
    )  # problem=None is ok for selection only

    # Extract indices of selected solutions
    # The selected population maintains the original order, so we need to find indices
    selected_indices = []
    for selected_ind in selected_pop:
        # Find matching individual in original population
        for i, orig_ind in enumerate(pop):
            if np.allclose(selected_ind.F, orig_ind.F):
                selected_indices.append(i)
                break

    return np.array(selected_indices)


def get_reference_directions(n_objectives, k):
    """
    Generate reference directions suitable for k selections.

    Parameters:
    -----------
    n_objectives : int
        Number of objectives
    k : int
        Target number of selections (used to estimate partitions)

    Returns:
    --------
    ref_dirs : np.ndarray
        Reference directions
    """
    # Estimate number of partitions needed
    # The number of reference directions grows combinatorially with partitions
    # Formula: C(n_objectives + p - 1, p) where p is partitions

    if n_objectives <= 2:
        # For 2D, partitions directly control number of directions
        n_partitions = k - 1
    else:
        # For higher dimensions, estimate partitions to get ~k directions
        # This is approximate; you may need to tune
        n_partitions = max(1, int(np.power(k, 1.0 / n_objectives)))

    # Generate uniform reference directions using Das-Dennis approach
    ref_dirs = UniformReferenceDirectionFactory(
        n_objectives, n_partitions=n_partitions
    ).do()

    return ref_dirs


# %% [markdown]
# ### Sample initial tasks


# %%
def sample_initial_tasks(
    rng, n, sample_random_function, random_sampling_config, get_key=None
):
    """
    rng: numpy rng
    n: number of initial random tasks
    sample_random_function: same as GAME
    random_sampling_config: same as GAME
    get_key: sol -> unique_identifier (if None do not check that that the initial solutions are different)
    """
    initial = []
    keys = set()
    while len(keys) < n:
        sol, _ = sample_random_function(rng, random_sampling_config)
        key = get_key(sol) if get_key else len(keys)
        if key not in keys:
            keys.add(key)
            initial.append(sol)
    return initial


def create_tasks(solutions, generation):
    tasks = []
    for i, s in enumerate(solutions):
        s["generation"] = generation
        tasks.append(s)
    return tasks


# %% [markdown]
# ### Main


# %%
class GAME:
    def __init__(self, config):
        self.config = config
        self.get_env_config = config["get_env_config"]
        self.budget = config["budget"]
        self.n_tasks = config["n_tasks"]
        self.n_cells = config["n_cells"]
        self.n_gen = config["n_gen"]
        self.seed_id = config["seed_id"]
        self.verbose = config["verbose"]
        self.batch_size = config["batch_size"]
        self.current_side = config["starting_side"]
        self.bootstrap_evaluations = []
        self.mtmb_mes = []
        self.update_current_side_MTMB_ME_config()
        if config["parallel"]:
            self.compute_tournament = compute_tournament_multi_proc
        else:
            self.compute_tournament = compute_tournament_in_batch
        assert config["tasks_selection"] in [
            "behavior",
            "fitness_quality",
            "ranking_quality",
            "fitness_diversity",
            "ranking_diversity",
            "pareto",
            "random",
        ]
        # if config["tasks_selection"] == "behavior":
        #     self.compute_next_generation = (
        #         self.compute_next_generation_with_behavior_diversity
        #     )
        # elif config["tasks_selection"] == "random":
        #     self.compute_next_generation = self.compute_next_generation_at_random
        # else:
        self.compute_next_generation = partial(
            self.compute_next_generation_with_tournament_diversity,
            config["tasks_selection"],
        )
        self.get_key = self.MTMB_ME_config["env_config"]["get_key"]
        self.main_folder = (
            config["main_folder"]
            if "main_folder" in config
            else utils.create_save_folder(verbose=self.verbose)
        )
        self.rng = np.random.default_rng(self.MTMB_ME_config["seed"])
        # Initialise the first set of tasks
        if config["initial_tasks"] is not None:
            assert len(config["initial_tasks"]) == self.n_tasks
            self.current_gen = config["initial_tasks"]
        else:
            task_env_config = self.get_env_config(
                "red" if self.current_side == "blue" else "blue", None
            )
            self.current_gen = sample_initial_tasks(
                self.rng,
                self.n_tasks,
                task_env_config["sample_random_function"],
                task_env_config["random_sampling_config"],
                self.get_key,
            )

    def update_current_side_MTMB_ME_config(self):
        self.MTMB_ME_config = get_MTMB_ME_config(
            self.current_side,
            get_env_config=self.get_env_config,
            budget=self.budget,
            n_cells=self.n_cells,
            batch_size=self.batch_size,
            init_elites=self.config["init_elites"],
            parallel=self.config["parallel"],
            n_proc=self.config["n_proc"],
            parallel_timeout=self.config["parallel_timeout"],
            archive_type=self.config["archive_type"],
            seed_id=self.seed_id,
            verbose=self.verbose,
            use_logging=self.config["use_logging"],
            log_interval=self.config["log_interval"],
            use_redristribution=self.config["use_redristribution"]
            if "use_redristribution" in self.config
            else False,
            use_collection=self.config["use_collection"]
            if "use_collection" in self.config
            else False,
            DNS_k=self.config["DNS_k"] if "DNS_k" in self.config else None,
        )

    def compute_next_generation_with_tournament_diversity(
        self, diversity_type, mtmb_me, gen_id
    ):
        assert diversity_type in [
            "behavior",
            "fitness_quality",
            "ranking_quality",
            "fitness_diversity",
            "ranking_diversity",
            "pareto",
        ]
        # compute tournament between all elites and current tasks (N_tasks**2*N_cells)
        elites = [
            elite
            for archive in mtmb_me.archive.archives
            for elite in archive.cells_solution.values()
        ]
        tasks = mtmb_me.tasks
        if self.current_side == "red":
            reds, blues = elites, tasks
        else:
            reds, blues = tasks, elites
        utils.save_pickle(
            self.MTMB_ME_config["save_folder"] + f"tournament_solutions_{gen_id}.pk",
            {"reds": reds, "blues": blues},
        )
        tournament = self.compute_tournament(self.MTMB_ME_config, reds, blues)
        utils.save_pickle(
            self.MTMB_ME_config["save_folder"] + f"tournament_evaluations_{gen_id}.pk",
            tournament,
        )

        # Select next generation of tasks
        F = np.zeros((len(reds), len(blues)))
        for (red_id, blue_id), ev in tournament.items():
            F[red_id, blue_id] = (
                ev["eval"]["fitness"]
                if self.current_side == "red"
                else ev["eval"]["other_fitness"]
            )
        F = F if self.current_side == "red" else F.T  # [n_elites, n_tasks]

        if "fitness" in diversity_type:
            criterion = diversity_type.split("_")[1]  # quality or diversity
            selected_elites_ids = select_new_tasks(F, self.n_tasks, F, criterion)
        elif "ranking" in diversity_type:
            criterion = diversity_type.split("_")[1]  # quality or diversity
            rankings = np.argsort(
                np.argsort(F, axis=1), axis=1
            )  # double argsort because we want to know at id (i,j) the rank of task j for elite i (one argsort would gives the id of the jth task)
            normalized_rankings = (
                rankings / (rankings.shape[1] / 2) - 1
            )  # normalized between -1 and 1 to use L2
            selected_elites_ids = select_new_tasks(
                normalized_rankings, self.n_tasks, F, criterion
            )
        else:
            selected_elites_ids = select_solutions_nsga3(
                -F, k=self.n_tasks
            )  # -F because NGSA-III minimizes

        self.current_gen = [elites[i] for i in selected_elites_ids]
        tasks_ids = {x: i for i, x in enumerate(selected_elites_ids)}
        return tournament, reds, blues, tasks_ids

    def run(self):
        for gen_id in range(self.n_gen):
            if gen_id > 0:
                self.current_side = (
                    "red" if self.current_side == "blue" else "blue"
                )  # switch side
            if self.verbose:
                print(
                    f"{self.current_side} gen {gen_id}: {len(set([self.get_key(sol) for sol in self.current_gen]))} different tasks."
                )
            self.update_current_side_MTMB_ME_config()
            self.MTMB_ME_config["save_folder"] = self.main_folder + f"/gen_{gen_id}/"
            utils.create_folder(self.MTMB_ME_config["save_folder"])
            self.MTMB_ME_config["tasks"] = create_tasks(
                self.current_gen, generation=self.current_side
            )  # gen == red means optimizing the reds and fixing the blues
            mtmb_me = MTMB_ME(self.MTMB_ME_config)
            self.mtmb_mes.append(mtmb_me)
            # bootstrap with last generation
            mtmb_me.update_archive(self.bootstrap_evaluations)
            # run current gen
            mtmb_me.run()
            # compute next gen
            tournament, reds, blues, tasks_ids = self.compute_next_generation(
                mtmb_me, gen_id
            )
            # save next gen
            utils.save_pickle(
                self.MTMB_ME_config["save_folder"] + f"new_tasks_{gen_id}.pk",
                {self.current_side: self.current_gen},
            )
            # compute the bootstrap evalution for next generation
            self.bootstrap_evaluations = []
            for (i, j), val in tournament.items():
                if self.current_side == "blue":  # blues
                    if j in tasks_ids:
                        f = val[
                            "eval"
                        ][
                            "fitness"
                        ]  # red fitness (because the tournament is always red candidates vs blue tasks)
                        b = val["eval"]["behavior"]
                        self.bootstrap_evaluations.append(
                            {
                                "id": -1,
                                "task_id": tasks_ids[j],
                                "fitness": f,
                                "behavior": b,
                                "solution": reds[i],
                            }
                        )
                else:
                    if i in tasks_ids:
                        f = val[
                            "eval"
                        ][
                            "other_fitness"
                        ]  # blue fitness (because the tournament is always red candidates vs blue tasks)
                        b = (
                            val["eval"]["other_behavior"]
                            if "other_behavior" in val["eval"]
                            else val["eval"]["behavior"]
                        )  # when the behavior is not shared between the two sides
                        self.bootstrap_evaluations.append(
                            {
                                "id": -1,
                                "task_id": tasks_ids[i],
                                "fitness": f,
                                "behavior": b,
                                "solution": blues[j],
                            }
                        )


# %% [markdown]
# ## Inter-Generational tournament


# %%
def compute_inter_generational_tournament(
    main_folder,
    n_gen,
    get_env_config,
    seed_id,
    batch_size,
    n_proc,
    compute_tournament,
    starting_side,
    log_level,
):
    Elites = [
        utils.load_pickle(main_folder + f"/gen_{gen_id}/new_tasks_{gen_id}.pk")
        for gen_id in range(n_gen)
    ]
    config = get_MTMB_ME_config(
        "red",
        get_env_config=get_env_config,
        budget=None,
        n_cells=None,
        batch_size=batch_size,
        init_elites=None,
        parallel=None,
        n_proc=n_proc,
        parallel_timeout=None,
        archive_type=None,
        seed_id=seed_id,
    )
    blue_gens = range(
        starting_side == "red", n_gen, 2
    )  # should be 0, 2, 4, ... if starting with blue
    red_gens = range(
        starting_side == "blue", n_gen, 2
    )  # should be 0, 2, 4, ... if starting with red
    # split the tournament in multiple save files because pickle can't open the one-save file if it's too big
    for blue_gen_id, red_gen_id in tqdm([(b, r) for b in blue_gens for r in red_gens]):
        mini_tournament = compute_tournament(
            config,
            Elites[red_gen_id]["red"],
            Elites[blue_gen_id]["blue"],
            verbose=False,
            log_level=log_level,
        )
        utils.save_pickle(
            main_folder + f"intergenerational_tournament_{blue_gen_id}_{red_gen_id}.pk",
            mini_tournament,
        )
