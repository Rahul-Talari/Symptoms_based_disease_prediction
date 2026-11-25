"""The `symptom-dx` command.

symptom-dx preprocess        raw CSV -> processed CSV + text corpus
symptom-dx build-kg          load the processed data into Neo4j
symptom-dx build-index       embed the corpus into the vector index
symptom-dx predict "..."     predict diseases for comma-separated symptoms
symptom-dx plot              redraw the evaluation charts
"""

import argparse
import sys

import pandas as pd

from symptom_dx import config
from symptom_dx.pipeline import SOURCES


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="symptom-dx", description="Symptom-based disease prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preprocess", help="raw CSV -> processed CSV + text corpus")

    build_kg = commands.add_parser("build-kg", help="load the processed data into Neo4j")
    build_kg.add_argument("--reset", action="store_true", help="delete everything in the database first")

    commands.add_parser("build-index", help="embed the corpus into the vector index")

    predict = commands.add_parser("predict", help="predict diseases for comma-separated symptoms")
    predict.add_argument("symptoms", help='e.g. "itching, skin rash, nodal skin eruptions"')
    predict.add_argument(
        "--sources", nargs="+", choices=SOURCES, default=list(SOURCES), help="sources to use (default: all)"
    )
    predict.add_argument("--show-sources", action="store_true", help="also print each source's own answer")
    predict.add_argument("--verbose", action="store_true", help="print the chain and agent steps")

    commands.add_parser("plot", help="redraw the evaluation charts")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.command == "preprocess":
        from symptom_dx.data import run_preprocessing

        df = run_preprocessing(config.RAW_DATA, config.PROCESSED_DATA, config.CORPUS)
        print(f"Wrote {len(df)} records for {df['prognosis'].nunique()} diseases.")

    elif args.command == "build-kg":
        from symptom_dx.knowledge_graph import build_graph, connect

        build_graph(connect(config.load_settings()), pd.read_csv(config.PROCESSED_DATA), reset=args.reset)
        print("Knowledge graph loaded.")

    elif args.command == "build-index":
        from symptom_dx.vector_store import build_index

        count = build_index(config.load_settings())
        print(f"Indexed {count} records into {config.VECTOR_INDEX_DIR}.")

    elif args.command == "predict":
        from symptom_dx.llm import get_chat_model
        from symptom_dx.pipeline import build_retrievers, predict

        settings = config.load_settings()
        llm = get_chat_model(settings)
        retrievers = build_retrievers(args.sources, settings, llm, verbose=args.verbose)
        result = predict(args.symptoms, retrievers, llm)

        print(f"Symptoms: {', '.join(result.symptoms)}\n")
        if args.show_sources:
            for name, answer in result.answers.items():
                print(f"## {name}\n{answer}\n")
        for name, error in result.errors.items():
            print(f"Warning: {name} failed: {error}", file=sys.stderr)
        print(result.report)

    elif args.command == "plot":
        from symptom_dx.plots import plot_inference_time, plot_metrics

        results = config.RESULTS_DIR
        plot_metrics(f"{results}/metrics.csv", f"{results}/figures/metrics_plot.png")
        plot_inference_time(f"{results}/inference_time.csv", f"{results}/figures/inference_plot.png")
        print(f"Charts saved in {results}/figures/.")


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        run(args)
    except (config.MissingSettingError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
