import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))   
from symb_xmil import QueryGenerator, run_symbolic_xmil

from toy_experiments.datasets import bag_collate_fn, get_number_mil_dataset
from toy_experiments.models import get_model_and_classifier
from toy_experiments.scripts import train_model
from toy_experiments.symbolic import ToyPerturbationScorer, toy_concepts


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--num-repetitions", type=int, default=1)

    parser.add_argument("--dataset-type", type=str, required=True)
    parser.add_argument("--num-numbers", type=int, default=10)
    def parse_num_instances(value):
        if "," in value:
            low, high = value.replace("(", "").replace(")", "").split(",")
            return (int(low.strip()), int(high.strip()))
        return int(value)


    parser.add_argument("--num-instances", type=parse_num_instances, default=30)
    parser.add_argument("--test-num-instances", type=int, default=30)
    parser.add_argument(
        "--sampling",
        type=str,
        default="hierarchical",
        choices=["unique", "uniform", "hierarchical", "pure"],
    )
    parser.add_argument(
        "--test-sampling",
        type=str,
        default="pure",
        choices=["unique", "uniform", "hierarchical", "pure"],
    )
    parser.add_argument(
        "--features-type",
        type=str,
        default="mnist",
        choices=[
            "onehot",
            "mnist",
            "mnist_resnet18",
            "mnist_resnet18_blur",
            "mnist_resnet18_color",
            "mnist_resnet18_marker",
            "cmnist_random_resnet18",
            "mnist_decoy",
        ],
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Directory with cached class_{0..9}.pt features; overrides the default location.",
    )
    parser.add_argument("--num-bags-train", type=int, default=2000)
    parser.add_argument("--num-bags-val", type=int, default=500)
    parser.add_argument("--num-bags-test", type=int, default=1000)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0)

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["attention_mil", "transmil"],
    )
    parser.add_argument("--model-dims", type=int, default=20)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--n-out-layers", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="best")

    parser.add_argument("--detach-pe", action="store_true")

    parser.add_argument("--run-symbolic-xmil", action="store_true")
    parser.add_argument("--symb-max-subset-size", type=int, default=None)
    parser.add_argument("--symb-max-query-size", type=int, default=3)
    parser.add_argument("--symb-explained-class", type=int, default=-1)
    parser.add_argument("--symb-save-mapped", action="store_true")

    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = get_args()

    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    save_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(save_dir)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    dataset_train = get_number_mil_dataset(
        dataset_type=args.dataset_type,
        num_numbers=args.num_numbers,
        num_bags=args.num_bags_train,
        num_instances=args.num_instances,
        features_type=args.features_type,
        sampling=args.sampling,
        noise=args.noise,
        threshold=args.threshold,
        features_path=args.features_path,
    )
    dataset_val = get_number_mil_dataset(
        dataset_type=args.dataset_type,
        num_numbers=args.num_numbers,
        num_bags=args.num_bags_val,
        num_instances=args.num_instances,
        features_type=args.features_type,
        sampling=args.sampling,
        noise=args.noise,
        threshold=args.threshold,
        features_path=args.features_path,
    )
    dataset_test = get_number_mil_dataset(
        dataset_type=args.dataset_type,
        num_numbers=args.num_numbers,
        num_bags=args.num_bags_test,
        num_instances=args.test_num_instances,
        features_type=args.features_type,
        sampling=args.test_sampling,
        noise=args.noise,
        threshold=args.threshold,
        features_path=args.features_path,
    )

    collate_fn = (
        bag_collate_fn
        if args.model_type in ["attention_mil", "additive_mil"]
        else None
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        collate_fn=collate_fn,
    )

    all_results = []

    for repetition in range(args.num_repetitions):
        model, classifier = get_model_and_classifier(
            model_type=args.model_type,
            num_features=dataset_train.num_features,
            num_classes=dataset_train.num_classes,
            model_dims=args.model_dims,
            dropout=args.dropout,
            n_out_layers=args.n_out_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
        )

        train_metrics = train_model(
            classifier=classifier,
            num_classes=dataset_train.num_classes,
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_val,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup=args.warmup,
            tolerance=args.tolerance,
            patience=args.patience,
            checkpoint=args.checkpoint,
        )

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{repetition}.pt"))

        test_metrics = {}

        if args.run_symbolic_xmil:
            symb_dir = os.path.join(save_dir, f"symb_xmil_{repetition}")
            concepts = toy_concepts(args.num_numbers)

            query_gen = QueryGenerator(
                max_query_size=args.symb_max_query_size,
                semantic_map="toy_experiment",
                negation=False,
            )
            scorer = ToyPerturbationScorer(
                classifier=classifier,
                explained_class=args.symb_explained_class,
                device=device,
            )

            symb_results = run_symbolic_xmil(
                scorer=scorer,
                dataloader=data_loader_test,
                results_dir=symb_dir,
                query_gen=query_gen,
                concepts=concepts,
                max_subset_size=args.symb_max_subset_size,
                verbose=True,
                save_mapped=args.symb_save_mapped,
            )

            test_metrics["symb_xmil"] = {
                "results_dir": symb_dir,
                "num_bags": int(symb_results["f_df"].shape[0]),
                "num_subsets": int(symb_results["f_df"].shape[1] - 1),
                "num_queries": int(symb_results["best_df"].shape[1] - 2),
            }

        all_metrics = {**train_metrics, "test": test_metrics}
        all_results.append(all_metrics)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()