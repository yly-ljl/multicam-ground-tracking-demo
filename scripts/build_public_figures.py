"""Draw publication SVGs from shipped summary numbers; no dataset images or models."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path, help='Optional local PNG QA directory, outside the publication tree')
    args = parser.parse_args()
    dest = ROOT / 'docs/assets'
    dest.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Microsoft YaHei' if 'Microsoft YaHei' in available else 'DejaVu Sans'
    plt.rcParams['svg.fonttype'] = 'path'
    plt.rcParams['axes.unicode_minus'] = False
    def save(fig, name):
        fig.savefig(dest / f'{name}.svg', bbox_inches='tight', facecolor='white')
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / f'{name}.png', dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14.5, 4.7))
    ax.set(xlim=(0, 14.5), ylim=(0, 4.6))
    ax.axis('off')
    ax.text(.1, 4.18, '固定三相机 → 米制地面轨迹', fontsize=20, fontweight='bold', color='#18364a')
    labels = [('原图', '读者自行取得'), ('YOLO11n检测', 'person框 · 底边中心'), ('脚点投地面', '固定标定 · 米制XY'),
              ('同帧融合', 'C1→C2→C3 · 等权'), ('时间跟踪', '贪心 · Kalman · d<1.5m'), ('展示与数值', '(X,Y,0) · 原图/BEV')]
    for i, (title, subtitle) in enumerate(labels):
        x = .1 + 2.38 * i
        ax.add_patch(FancyBboxPatch((x, 2.35), 2.15, 1.05, boxstyle='round,pad=.07', facecolor='#e9f2f5', edgecolor='#416a7c', linewidth=1.3))
        ax.text(x + 1.075, 3.03, title, ha='center', fontsize=12, fontweight='bold', color='#18364a')
        ax.text(x + 1.075, 2.64, subtitle, ha='center', fontsize=8.7, color='#385469')
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 2.2, 2.87), (x + 2.34, 2.87), arrowstyle='-|>', mutation_scale=14, color='#416a7c'))
    ax.add_patch(FancyBboxPatch((4.1, .65), 3.05, 1.0, boxstyle='round,pad=.07', facecolor='#e5f4ee', edgecolor='#288566'))
    ax.text(5.625, 1.28, '已有XY预测缓存', ha='center', fontsize=12, color='#19543f')
    ax.text(5.625, .91, 'replay跳过检测与投影', ha='center', fontsize=10, color='#19543f')
    ax.add_patch(FancyArrowPatch((7.2, 1.25), (8.1, 2.3), arrowstyle='-|>', mutation_scale=14, color='#288566'))
    ax.add_patch(FancyBboxPatch((9.0, .65), 5.1, 1.0, boxstyle='round,pad=.08', facecolor='#fff5e5', edgecolor='#ae772b'))
    ax.text(11.55, 1.28, 'GT只进入事后空间 / 整段身份评价与诊断', ha='center', fontsize=11, color='#76501f')
    ax.text(11.55, .91, '1m评价门 ≠ 0.75m融合门 ≠ d<1.5m时间门', ha='center', fontsize=9.5, color='#76501f')
    ax.text(.12, .03, '默认B3；原图/标定/权重由读者自行取得。E007诊断标签不进入预测。', fontsize=10, color='#536674')
    save(fig, 'pipeline')

    base = load(ROOT / 'results/summary/E004_identity.csv')
    global_rows = load(ROOT / 'results/summary/E005_identity.csv')
    soft = load(ROOT / 'results/summary/E006_identity.csv')
    segments = ['old_development', 'new_development', 'seen_reference']
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    variants = [('B2', base, '#264c66'), ('B3 默认', base, '#0b8b8b'), ('GLOBAL 未采用', global_rows, '#aeb9c0'), ('SOFT 未采用', soft, '#d99b54')]
    x = np.arange(3); width = .18
    for j, (label, data, color) in enumerate(variants):
        variant = ['B2', 'B3', 'B3_GLOBAL', 'B3_SOFT'][j]
        values = [100 * float(next(r for r in data if r['segment'] == s and r['variant'] == variant)['idf1']) for s in segments]
        positions = x + (j - 1.5) * width
        ax.bar(positions, values, width, color=color, label=label)
        for at, value in zip(positions, values):
            ax.text(at, value + 1.2, f'{value:.2f}', ha='center', fontsize=8.5, color='#213a4a')
    ax.set_xticks(x, ['旧开发 0–495', '新开发 500–795', '已见参考 800–1395'])
    ax.set_ylim(0, 100);ax.set_ylabel('地面点 IDF1@1m (%)')
    ax.set_title('三路增加覆盖，身份分数仍低于双路；两个时间候选未采用', fontsize=13, pad=17)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, .94), frameon=False, fontsize=9)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y', alpha=.17);ax.set_axisbelow(True)
    fig.text(.12, -.015, '全ROI · filtered_observed · 各段独立整段对应；不是官方WILDTRACK完整成绩。', fontsize=9, color='#536674')
    fig.tight_layout();save(fig, 'identity_comparison')
    print('Wrote two self-drawn SVGs from shipped numbers/code relationships.')


if __name__ == '__main__':
    main()
