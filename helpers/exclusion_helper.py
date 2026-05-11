"""Helper for generating subject inclusion/exclusion bar charts (AA database)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExclusionHelper:
    """Computes and plots included vs. excluded subjects per age group."""

    GROUPS_BY_AGE_SETTING = {
        20: ['18-29y', '30-49y', '50+y'],
        10: ['18-29y', '30-39y', '40-49y', '50-59y', '60-69y', '70-79y', '80-92y'],
    }
    AGE_CODE_TO_20Y = {
        1: '18-29y', 2: '18-29y', 3: '18-29y',
        4: '30-49y', 5: '30-49y', 6: '30-49y', 7: '30-49y',
        8: '50+y',   9: '50+y',  10: '50+y',  11: '50+y',
       12: '50+y',  13: '50+y',  14: '50+y',  15: '50+y',
    }

    COLOR_INCLUDED = '#4393c3'
    COLOR_EXCLUDED = '#d6604d'
    BAR_WIDTH = 0.35
    FONT_SIZE = 14
    LEGEND_FONT_SIZE = 14
    TITLE_FONT_SIZE = 14

    def __init__(self, results_path: str, subject_info_path: str, age_group: int = 20):
        """
        Args:
            results_path: Path to population_results_autonomic_aging(*yGroups).csv
            subject_info_path: Path to subject_info_aa.csv
            age_group: 20 or 10 — controls group labels and subject_info column used
        """
        self.results_path = results_path
        self.subject_info_path = subject_info_path
        self.age_group = age_group
        self.groups = self.GROUPS_BY_AGE_SETTING[age_group]
        self._counts = self._compute_counts()

    def _compute_counts(self) -> pd.DataFrame:
        """Count total, included, and excluded subjects per age group."""
        results = pd.read_csv(self.results_path)
        subj = pd.read_csv(self.subject_info_path)

        if self.age_group == 20:
            subj['_group'] = subj['Age_group'].map(self.AGE_CODE_TO_20Y)
        else:
            subj['_group'] = subj['population_group_10_years']

        total = (
            subj.groupby('_group')['ID']
            .nunique()
            .reindex(self.groups, fill_value=0)
        )
        included = (
            results.groupby('population_group')['control_number']
            .nunique()
            .reindex(self.groups, fill_value=0)
        )
        return pd.DataFrame({
            'total':    total,
            'included': included,
            'excluded': total - included,
        })

    def save_exclusion_barchart(
        self, output_dir: str, filename: str = 'exclusion_barchart.png'
    ):
        """Save a grouped bar chart of included vs. excluded subjects per age group."""
        os.makedirs(output_dir, exist_ok=True)

        included_vals = self._counts['included'].values
        excluded_vals = self._counts['excluded'].values
        x = np.arange(len(self.groups))
        w = self.BAR_WIDTH

        _, ax = plt.subplots(figsize=(9, 6))

        included_bars = ax.bar(
            x - w / 2, included_vals, width=w,
            color=self.COLOR_INCLUDED, label='Included', zorder=3,
        )
        excluded_bars = ax.bar(
            x + w / 2, excluded_vals, width=w,
            color=self.COLOR_EXCLUDED, label='Excluded', zorder=3,
        )

        total_vals = self._counts['total'].values
        for bars, vals in [(included_bars, included_vals), (excluded_bars, excluded_vals)]:
            for rect, group_total in zip(bars, total_vals):
                count = int(rect.get_height())
                pct = 100 * count / group_total if group_total > 0 else 0
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 2,
                    f'{count}({pct:.0f}%)',
                    ha='center', va='bottom',
                    fontsize=self.FONT_SIZE - 1, fontweight='bold',
                )

        ax.set_xticks(x)
        ax.set_xticklabels(self.groups, fontsize=self.FONT_SIZE)
        ax.set_ylabel('Number of subjects', fontsize=self.FONT_SIZE)
        ax.set_xlabel('Age group', fontsize=self.FONT_SIZE)
        ax.set_title(
            'Subjects included vs. excluded per age group\n(Autonomic Ageing database)',
            fontsize=self.TITLE_FONT_SIZE,
        )
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=self.LEGEND_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=self.FONT_SIZE - 1)

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved exclusion barchart → {out_path}')
