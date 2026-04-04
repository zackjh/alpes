from pathlib import Path
from model_wrapper import F3EDTennis
from datapool import DataPool
from sampler import RandomSampler
from utils import import_data_from_json, export_data_to_json


# Experiment config
EXPERIMENT_NAME = "uncertainty_sampling_top_1000"

F3SET_REPO_ROOT = Path("/mnt/ssd2/zachary/alpes/f3set")

SEED = 1234

INITIAL_LABELED_POOL_SIZE = 1000
QUERY_BATCH_SIZE = 1000

NUM_OF_EPOCHS = 25
START_VAL_EPOCH = 15

print(f"[LOG][active_learning_experiment.py] Experiment Name is: {EXPERIMENT_NAME}")
print(f"[LOG][active_learning_experiment.py] Seed is: {SEED}")

# Get absolute path for experiment results (i.e. not a relative path)
experiment_results_path = Path(EXPERIMENT_NAME).resolve()

# Create directory to store experiment results
# If the directory already exists, throw a FileExistsError
experiment_results_path.mkdir(parents=True, exist_ok=False)

all_training_data = import_data_from_json(
    F3SET_REPO_ROOT / "data" / "f3set-tennis" / "train.json"
)  # Get the full training data set
labeled_pool = DataPool([])  # Labeled pool starts off as an empty list
unlabeled_pool = DataPool(all_training_data)


random_sampler = RandomSampler(SEED)

model = F3EDTennis(F3SET_REPO_ROOT)

# Move initial batch from unlabeled pool to labeled pool
initial_batch = random_sampler.get_samples(INITIAL_LABELED_POOL_SIZE, unlabeled_pool)
unlabeled_pool.remove_samples(initial_batch)
labeled_pool.add_samples(initial_batch)


# Keep track of how many active learning iterations occur
active_learning_iteration = 0  # pylint: disable=C0103

# Run active learning loop
while True:
    # Get path for the current active learning iteration
    active_learning_iteration_path = (
        experiment_results_path
        / f"active_learning_iteration_{active_learning_iteration}"
    )

    # Create directory for the current active learning iteration
    active_learning_iteration_path.mkdir(parents=True, exist_ok=False)

    # Get path for the training data for the current active learning iteration
    active_learning_iteration_train_json_path = (
        active_learning_iteration_path / "train.json"
    )

    # Export data from labeled pool into JSON
    export_data_to_json(active_learning_iteration_train_json_path, labeled_pool)

    # Logging for my own sanity
    print(
        f"[LOG][active_learning_experiment.py] Active Learning Iteration {active_learning_iteration}"
    )
    print(
        f"[LOG][active_learning_experiment.py] Size of Unlabeled Pool: {unlabeled_pool.size()}"
    )
    print(
        f"[LOG][active_learning_experiment.py] Size of Labeled Pool: {labeled_pool.size()}"
    )
    print(
        f"[LOG][active_learning_experiment.py] Training model on {len(labeled_pool.data)} samples"
    )

    # Train model
    model.train(
        results_path=active_learning_iteration_path,
        num_of_epochs=NUM_OF_EPOCHS,
        start_val_epoch=START_VAL_EPOCH,
        train_json_path=active_learning_iteration_train_json_path,
    )

    # All samples have been used - end experiment
    if unlabeled_pool.size() == 0:
        break

    # Export unlabeled data into `unlabeled_data.json`
    unlabeled_data_json_path = active_learning_iteration_path / "unlabeled_data.json"
    export_data_to_json(unlabeled_data_json_path, unlabeled_pool)

    # Get query batch
    query_batch_video_names = model.get_query_batch_video_names(
        active_learning_iteration_path, unlabeled_data_json_path, QUERY_BATCH_SIZE
    )

    # Remove query batch from the unlabeled pool
    query_batch = unlabeled_pool.remove_samples_from_list_of_video_names(
        query_batch_video_names
    )

    # Add query batch to the labeled pool
    labeled_pool.add_samples(query_batch)

    active_learning_iteration += 1
