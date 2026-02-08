#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(B, N):
    fname = os.path.join(DATA_DIR, f'data_B{B}_N{N}.csv')
    if not os.path.exists(fname):
        return None
    return pd.read_csv(fname)


def plot1_single_stream(B=14, N=500000, stream_id=0):
    df = load_data(B, N)
    if df is None:
        print(f"No data for B={B}, N={N}")
        return

    single = df[df['stream_id'] == stream_id]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(single['step'], single['exact_f0'], 'b-', linewidth=2, label=r'$F_0^t$ (точное)', marker='o', markersize=4)
    ax.plot(single['step'], single['hll_estimate'], 'r--', linewidth=2, label=r'$N_t$ (HyperLogLog)', marker='s', markersize=4)
    ax.set_xlabel('Размер обработанной части потока')
    ax.set_ylabel('Количество уникальных элементов')
    ax.set_title(f'Сравнение $F_0^t$ и $N_t$ (B={B}, m={2**B}, поток #{stream_id})')
    ax.legend(fontsize=11)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot1_B{B}_N{N}.png'), dpi=150)
    plt.close()
    print(f"Saved plot1_B{B}_N{N}.png")


def plot1_multiple_B(N=500000, stream_id=0):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    B_values = [4, 8, 10, 14]

    for ax, B in zip(axes.flat, B_values):
        df = load_data(B, N)
        if df is None:
            continue
        single = df[df['stream_id'] == stream_id]
        ax.plot(single['step'], single['exact_f0'], 'b-', linewidth=2, label=r'$F_0^t$', marker='o', markersize=3)
        ax.plot(single['step'], single['hll_estimate'], 'r--', linewidth=1.5, label=r'$N_t$ (HLL)', marker='s', markersize=3)
        ax.set_title(f'B={B}, m={2**B}')
        ax.set_xlabel('Элементов обработано')
        ax.set_ylabel('Уникальных')
        ax.legend(fontsize=9)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

    plt.suptitle(f'График №1: сравнение $F_0^t$ и $N_t$ для разных B (N={N})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot1_multi_B_N{N}.png'), dpi=150)
    plt.close()
    print(f"Saved plot1_multi_B_N{N}.png")


def plot2_statistics(B=14, N=500000):
    df = load_data(B, N)
    if df is None:
        print(f"No data for B={B}, N={N}")
        return

    grouped = df.groupby('step').agg(
        mean_exact=('exact_f0', 'mean'),
        mean_hll=('hll_estimate', 'mean'),
        std_hll=('hll_estimate', 'std'),
        mean_hllp=('hllp_estimate', 'mean'),
        std_hllp=('hllp_estimate', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(grouped['step'], grouped['mean_hll'], 'r-', linewidth=2, label=r'$\mathbb{E}(N_t)$ HLL')
    ax.fill_between(grouped['step'],
                     grouped['mean_hll'] - grouped['std_hll'],
                     grouped['mean_hll'] + grouped['std_hll'],
                     alpha=0.2, color='red', label=r'$\mathbb{E}(N_t) \pm \sigma_{N_t}$ HLL')

    ax.plot(grouped['step'], grouped['mean_exact'], 'b-', linewidth=2, label=r'$\mathbb{E}(F_0^t)$ (точное)')

    ax.set_xlabel('Размер обработанной части потока')
    ax.set_ylabel('Количество уникальных элементов')
    ax.set_title(f'Статистики оценки HyperLogLog (B={B}, m={2**B}, {df["stream_id"].nunique()} потоков)')
    ax.legend(fontsize=11)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot2_B{B}_N{N}.png'), dpi=150)
    plt.close()
    print(f"Saved plot2_B{B}_N{N}.png")


def plot2_multiple_B(N=500000):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    B_values = [4, 8, 10, 14]

    for ax, B in zip(axes.flat, B_values):
        df = load_data(B, N)
        if df is None:
            continue

        grouped = df.groupby('step').agg(
            mean_exact=('exact_f0', 'mean'),
            mean_hll=('hll_estimate', 'mean'),
            std_hll=('hll_estimate', 'std'),
        ).reset_index()

        ax.plot(grouped['step'], grouped['mean_exact'], 'b-', linewidth=2, label=r'$F_0^t$')
        ax.plot(grouped['step'], grouped['mean_hll'], 'r-', linewidth=1.5, label=r'$\mathbb{E}(N_t)$')
        ax.fill_between(grouped['step'],
                         grouped['mean_hll'] - grouped['std_hll'],
                         grouped['mean_hll'] + grouped['std_hll'],
                         alpha=0.2, color='red', label=r'$\pm\sigma$')
        ax.set_title(f'B={B}, m={2**B}')
        ax.set_xlabel('Элементов')
        ax.set_ylabel('Уникальных')
        ax.legend(fontsize=8)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

    plt.suptitle(f'График №2: $\\mathbb{{E}}(N_t) \\pm \\sigma$ для разных B (N={N})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot2_multi_B_N{N}.png'), dpi=150)
    plt.close()
    print(f"Saved plot2_multi_B_N{N}.png")


def plot_relative_error(N=500000):
    B_values = [4, 8, 10, 14]
    fig, ax = plt.subplots(figsize=(10, 6))

    empirical_stds = []
    theoretical_stds_104 = []
    theoretical_stds_130 = []
    m_values = []

    for B in B_values:
        df = load_data(B, N)
        if df is None:
            continue

        max_step = df['step'].max()
        final = df[df['step'] == max_step]

        rel_err = (final['hll_estimate'] - final['exact_f0']) / final['exact_f0']
        empirical_stds.append(rel_err.std())
        theoretical_stds_104.append(1.04 / np.sqrt(2**B))
        theoretical_stds_130.append(1.30 / np.sqrt(2**B))
        m_values.append(2**B)

    ax.plot(m_values, empirical_stds, 'ro-', linewidth=2, markersize=8, label=r'$\sigma_{emp}$ (эмпирическая)')
    ax.plot(m_values, theoretical_stds_104, 'b--', linewidth=2, markersize=8, label=r'$1.04/\sqrt{m}$ (теория)')
    ax.plot(m_values, theoretical_stds_130, 'g:', linewidth=2, markersize=8, label=r'$1.30/\sqrt{m}$ (верхняя граница)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('m (число регистров)')
    ax.set_ylabel('Относительное стд. отклонение')
    ax.set_title(f'Сравнение эмпирической и теоретической ошибки (N={N})')
    ax.legend(fontsize=11)
    ax.set_xticks(m_values)
    ax.set_xticklabels([str(m) for m in m_values])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot_relerr_N{N}.png'), dpi=150)
    plt.close()
    print(f"Saved plot_relerr_N{N}.png")


def plot_hll_vs_hllplus(B=14, N=500000):
    df = load_data(B, N)
    if df is None:
        return

    grouped = df.groupby('step').agg(
        mean_exact=('exact_f0', 'mean'),
        mean_hll=('hll_estimate', 'mean'),
        std_hll=('hll_estimate', 'std'),
        mean_hllp=('hllp_estimate', 'mean'),
        std_hllp=('hllp_estimate', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(grouped['step'], grouped['mean_exact'], 'b-', linewidth=2, label=r'$F_0^t$ (точное)')
    ax.plot(grouped['step'], grouped['mean_hll'], 'r--', linewidth=1.5, label=r'$\mathbb{E}(N_t)$ HLL')
    ax.fill_between(grouped['step'],
                     grouped['mean_hll'] - grouped['std_hll'],
                     grouped['mean_hll'] + grouped['std_hll'],
                     alpha=0.15, color='red')
    ax.plot(grouped['step'], grouped['mean_hllp'], 'g-.', linewidth=1.5, label=r'$\mathbb{E}(N_t)$ HLL++')
    ax.fill_between(grouped['step'],
                     grouped['mean_hllp'] - grouped['std_hllp'],
                     grouped['mean_hllp'] + grouped['std_hllp'],
                     alpha=0.15, color='green')

    ax.set_xlabel('Размер обработанной части потока')
    ax.set_ylabel('Количество уникальных элементов')
    ax.set_title(f'Сравнение HLL и HLL++ (B={B}, N={N})')
    ax.legend(fontsize=11)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'plot_comparison_B{B}_N{N}.png'), dpi=150)
    plt.close()
    print(f"Saved plot_comparison_B{B}_N{N}.png")


if __name__ == '__main__':
    for N in [100000, 500000]:
        for B in [14]:
            plot1_single_stream(B=B, N=N)
            plot2_statistics(B=B, N=N)

    for N in [500000]:
        plot1_multiple_B(N=N)
        plot2_multiple_B(N=N)
        plot_relative_error(N=N)
        plot_hll_vs_hllplus(B=14, N=N)

    print("\nAll plots generated!")
