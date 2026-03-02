import matplotlib.pyplot as plt
import csv

epochs = []
losses = []
accuracies = []

try:
    with open('training_metrics.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            losses.append(float(row['Avg Loss']))
            accuracies.append(float(row['Accuracy']))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Avg Loss', color=color)
    ax1.plot(epochs, losses, color=color, label='Avg Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, accuracies, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training Metrics over Epochs')
    plt.grid(True)
    plt.savefig('metrics_plot.png')
    print("Plot saved to metrics_plot.png")
    plt.show()
except FileNotFoundError:
    print("training_metrics.csv not found. Please run train.py first.")
