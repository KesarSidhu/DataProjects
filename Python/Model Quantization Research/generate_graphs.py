import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# change the fiel paths here-- to your csv files  
base_df = pd.read_csv("base_model_report.csv")
dynamic_df= pd.read_csv("dynamic_quant_model_report.csv")
float16_df=pd.read_csv("float16_model_report.csv")
full_df=pd.read_csv("full_integer_model_report.csv")


# change the model_paths here-- to your model_paths file
model_paths = {
    "Base Model (H5)": "mobilenet_mnist.h5",
    "Dynamic Quantized (TFLite)": "mobilenet_dynamic_quantized.tflite",
    "Float16 Quantized (TFLite)": "mnist_model_quant_f16.tflite",
    "Full Int (TFLite)": "mobilenet_mnist_full_int.tflite"
}

models = [
    (base_df, "Base Model"),
    (dynamic_df, "Dynamic Range Model"),
    (float16_df, "Float 16 Model"),
    (full_df, "Full Integer Model")
]

# whatever outputfolder you choose 
output_folder='graphs'

def generate_accuracy_graph(models, output_path):
    # Extract accuracy from each DataFrame (third-last row, second column)
    accuracies = {
        model_name: df.iloc[-3, 1]
        for df, model_name in models
    }

    plt.figure(figsize=(8, 5))
    colors = ["blue", "green", "red", "orange"]
    bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)

    plt.ylabel("Accuracy")
    plt.ylim(0.9, 1.0)
    plt.title("Accuracy Comparison Across Models")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002,
                 f"{yval:.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Save figure
    save_path = os.path.join(output_path, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Call using your models list
generate_accuracy_graph(models, output_folder)

# # This will give a metric comparison across the models (precision, recall, f1-score)
def generate_grouped_metric_graph(metric, models, output_path):
    cleaned_dfs = {}

    # Filter and convert each DataFrame
    for df, name in models:
        df = df.copy()
        df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
        cleaned_dfs[name] = df

    # Setup for grouped bar plot
    bar_width = 0.2
    classes = cleaned_dfs["Base Model"].iloc[:, 0].astype(str)
    x = np.arange(len(classes))

    model_colors = {
        "Base Model": "blue",
        "Dynamic Range Model": "green",  # <- match this to your label
        "Float 16 Model": "red",
        "Full Integer Model": "orange"
    }

    plt.figure(figsize=(14, 6))
    for i, (label, df) in enumerate(cleaned_dfs.items()):
        plt.bar(x + i * bar_width, df[metric].values, width=bar_width,
                label=label, color=model_colors[label])

    plt.xticks(x + bar_width * 1.5, classes)
    plt.xlabel("Class")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} Comparison Across Models")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    save_path = os.path.join(output_path, f"{metric}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


metrics = ["precision", "recall", "f1-score"]

for metric in metrics:
    generate_grouped_metric_graph(metric, models, output_folder)


# This will plot the classification report for each indivudauls model 
def plot_classification_report(df, output_path, model_type):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Keep only rows with numeric class labels (0–9)
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]
    
    # Convert metric columns to numeric
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Plotting
    plt.figure(figsize=(12, 6))
    df_plot = df.set_index(df.columns[0])[["precision", "recall", "f1-score"]]
    df_plot.plot(kind="bar", colormap="viridis", ax=plt.gca())

    plt.title(f"Precision, Recall, and F1-Score for {model_type}")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Save figure
    filename = f"{model_type.replace(' ', '_')}.png"
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


for df, name in models:
    plot_classification_report(df, output_folder, name)

# define model paths s
def plot_model_size_comparison(model_paths, output_path):
    # Get file sizes in MB
    model_sizes = {name: os.path.getsize(path) / (1024 * 1024)
                   for name, path in model_paths.items()}

    colors = ["blue", "green", "red", "orange"]

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_sizes.keys(), model_sizes.values(), color=colors)

    plt.ylabel("Size (MB)")
    plt.title("Model Size Comparison")

    # Add size labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f} MB",
                 ha="center", fontsize=11, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15)

    # Save figure
    save_path = os.path.join(output_path, "model_size_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Call the function
plot_model_size_comparison(model_paths, output_folder)
