import json
import os

import matplotlib.pyplot as plt
import numpy as np

json_data = {}


def plot_evaluation_metrics(
    *json_inputs, title="Evaluation Metrics", fig_name="evaluation_metrics.png"
):
    metrics = ["avg_grammar", "avg_creativity", "avg_consistency", "avg_plot_sense"]

    num_datasets = len(json_inputs)

    plt.figure(figsize=(12, 6))

    bar_width = 0.7 / num_datasets

    positions = np.arange(len(metrics))

    colors = ["blue", "blue", "red", "red", "green", "green"]

    for i, (name, data) in enumerate(json_inputs):
        offset_positions = (
            positions + i * bar_width - (num_datasets - 1) * bar_width / 2
        )

        values = [data.get(metric, 0) for metric in metrics]

        plt.bar(
            offset_positions,
            values,
            width=bar_width,
            label=name,
            color=colors[i % len(colors)],
        )
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(positions, metrics, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(fig_name)
    plt.show()


def load_data(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                variable_name = os.path.splitext(filename)[0]
                json_data[variable_name] = data
                globals()[variable_name] = data


def main():
    folder_path = "data/normal_evals"
    load_data(folder_path)
    # folder_path = "data/contrastive_evals"
    # load_data(folder_path)

    global json_data

    for name, content in json_data.items():
        print(f"Variable: {name}")
        print(f"Content: {content}\n")

    names_to_evaluate = [
        "gpt2_128_8_eval",
        "comp_128_8_eval",
        "gpt2_256_8_eval",
        "comp_256_8_eval",
        "gpt2_512_8_eval",
        "comp_512_8_eval",
    ]
    json_data = {name: json_data[name] for name in names_to_evaluate}

    plot_evaluation_metrics(
        *json_data.items(),
        title="Model Evaluation Metrics Comparison",
        fig_name="compression.png",
    )


if __name__ == "__main__":
    main()
