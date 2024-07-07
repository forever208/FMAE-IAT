import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats



def F1_figure():
    # MAE pretraining on BP4D
    model_size = ['ViT-small', 'ViT-base', 'ViT-large']
    F1_imagenet_pretrain = [None, 64.5, 65.7]
    F1_face_pretrain = [63.0, 65.4, 66.6]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot(model_size, F1_imagenet_pretrain, linewidth=2, label='ImageNet pretraining (MAE)', color='indianred',
            alpha=0.9, marker='^', markersize=5,
            linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot(model_size, F1_face_pretrain, linewidth=2, label='Face8M pretraining (FMAE)', color='ForestGreen',
            alpha=0.9, marker='o', markersize=5,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    # plt.xlim(1, 21)
    plt.ylim(62, 67)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel("F1 score", fontsize=15)
    # plt.xlabel("timestep", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=13)  # loc='upper right'

    # Show the plot
    plt.savefig(
        'MAE_scale_effect.pdf',
        # bbox_inches='tight',
    )
    plt.show()


def Linear_prob_figure():
    # MAE pretraining on BP4D
    epoch = [i for i in range(1, 20, 2)]
    ID_acc_without_adversarial = [54.14, 66.91, 74.14, 77.64, 79.75, 81.13, 82.27, 82.76, 82.92, 83.08]
    ID_acc_with_adversarial =    [4.63, 14.55, 17.07, 19.35, 23.25, 23.17, 25.44, 25.77, 27.48, 27.88]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot(epoch, ID_acc_without_adversarial, linewidth=2, label='without IAT', color='indianred',
            alpha=0.9, marker='^', markersize=5,
            linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot(epoch, ID_acc_with_adversarial, linewidth=2, label='with IAT', color='ForestGreen',
            alpha=0.9, marker='o', markersize=5,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    # plt.xlim(1, 21)
    # plt.ylim(62, 67)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel("ID accuray", fontsize=15)
    plt.xlabel("epoch", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)  # loc='upper right'

    # Show the plot
    plt.savefig(
        'ID_linear_prob.pdf',
        # bbox_inches='tight',
    )
    plt.show()


def learning_dynamics_figure():
    # MAE pretraining on BP4D
    epoch = [i for i in range(1, 20, 2)]
    F1_without_adversarial = [61.74, 65.33, 64.08, 64.27, 63.01, 63.20, 63.25, 63.10, 63.00, 63.02]
    F1_with_adversarial =    [50.20, 51.58, 59.59, 62.67, 63.67, 65.04, 65.58, 66.66, 66.06, 66.26]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot(epoch, F1_without_adversarial, linewidth=2, label='without IAT', color='indianred',
            alpha=0.9, marker='^', markersize=5,
            linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot(epoch, F1_with_adversarial, linewidth=2, label='with IAT', color='ForestGreen',
            alpha=0.9, marker='o', markersize=5,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    # plt.xlim(1, 21)
    # plt.ylim(62, 67)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel("F1 score", fontsize=15)
    plt.xlabel("epoch", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)  # loc='upper right'

    # Show the plot
    plt.savefig(
        'F1_learning_dynamics.pdf',
        # bbox_inches='tight',
    )
    plt.show()


if __name__ == '__main__':
    # F1_figure()
    Linear_prob_figure()
    # learning_dynamics_figure()