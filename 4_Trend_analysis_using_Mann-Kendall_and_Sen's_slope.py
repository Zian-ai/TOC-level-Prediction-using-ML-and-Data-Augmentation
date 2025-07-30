import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
import os


# ─── Use Times New Roman everywhere ──────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.default'] = 'regular'  # ensure math text also uses Times


# **List of Prediction Files **(SACHEON RESERVOIR)
files = [
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\11-ID4383_SQM_ACCESS-CM2_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\11-ID4383_SQM_ACCESS-CM2_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\22-ID4383_SQM_ACCESS-ESM1-5_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\22-ID4383_SQM_ACCESS-ESM1-5_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\33-ID4383_SQM_CanESM5_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\33-ID4383_SQM_CanESM5_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\44-ID4383_SQM_CMCC-ESM2_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\44-ID4383_SQM_CMCC-ESM2_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\55-ID4383_SQM_TaiESM1_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\55-ID4383_SQM_TaiESM1_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\66-ID4383_SQM_NorESM2-MM_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\66-ID4383_SQM_NorESM2-MM_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\77-ID4383_SQM_NorESM2-LM_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\77-ID4383_SQM_NorESM2-LM_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\88-ID4383_SQM_NESM3_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\88-ID4383_SQM_NESM3_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\99-ID4383_SQM_MRI-ESM2-0_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\99-ID4383_SQM_MRI-ESM2-0_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\100-ID4383_SQM_MIROC6_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\100-ID4383_SQM_MIROC6_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\111-ID4383_SQM_KIOST-ESM_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\111-ID4383_SQM_KIOST-ESM_ssp585_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\112-ID4383_SQM_FGOALS-g3_ssp245_summary_predictions.csv",
    r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULT-SACHEON2\112-ID4383_SQM_FGOALS-g3_ssp585_summary_predictions.csv"
]

# # **List of Prediction Files** **(DOCHEON RESERVOIR)
# files = [
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\11-ID4981_SQM_ACCESS-CM2_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\11-ID4981_SQM_ACCESS-CM2_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\22-ID4981_SQM_ACCESS-ESM1-5_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\22-ID4981_SQM_ACCESS-ESM1-5_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\33-ID4981_SQM_CanESM5_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\33-ID4981_SQM_CanESM5_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\44-ID4981_SQM_CMCC-ESM2_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\44-ID4981_SQM_CMCC-ESM2_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\55-ID4981_SQM_TaiESM1_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\55-ID4981_SQM_TaiESM1_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\66-ID4981_SQM_NorESM2-MM_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\66-ID4981_SQM_NorESM2-MM_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\77-ID4981_SQM_NorESM2-LM_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\77-ID4981_SQM_NorESM2-LM_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\88-ID4981_SQM_NESM3_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\88-ID4981_SQM_NESM3_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\99-ID4981_SQM_MRI-ESM2-0_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\99-ID4981_SQM_MRI-ESM2-0_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\100-ID4981_SQM_MIROC6_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\100-ID4981_SQM_MIROC6_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\111-ID4981_SQM_KIOST-ESM_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\111-ID4981_SQM_KIOST-ESM_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\112-ID4981_SQM_FGOALS-g3_ssp245_summary_predictions(2-도천-BOOSTED-M5).csv",
#     r"C:\Climate_change_data\기후변화시나리오\CMIP6\11111-TESTING\RESULTSS-DOCHEON2\112-ID4981_SQM_FGOALS-g3_ssp585_summary_predictions(2-도천-BOOSTED-M5).csv"      
# ]

# Ensure 'files' is not empty before extracting model names
if files:
    models = sorted(set([os.path.basename(f).split("_")[2] for f in files]))
else:
    models = []

# **Create Dataframe to Store Results**
results = []

# --------------------------------------------------------------------
# HELPER FUNCTION: mann_kendall_test
def mann_kendall_test(years, data):
    years = np.array(years)
    data = np.array(data)
    if len(data) < 10 or len(years) < 10:
        return None, None, None, None, None
    mk_result = mk.original_test(data)
    trend_direction = mk_result.trend
    p_value = mk_result.p
    z_score = mk_result.z
    sen_slope = mk_result.slope
    return trend_direction, p_value, z_score, sen_slope, sen_slope

# --------------------------------------------------------------------
# HELPER FUNCTION: mann_kendall_sen_slope
def mann_kendall_sen_slope(years, data):
    years = np.array(years)
    data = np.array(data)
    if len(data) < 10:
        return None
    slopes = [(data[j] - data[i]) / (years[j] - years[i])
              for i in range(len(data)) 
              for j in range(i + 1, len(data))]
    sen_slope = np.median(slopes)
    intercept = np.median(data) - sen_slope * np.median(years)
    trend = sen_slope * years + intercept
    return trend

# --------------------------------------------------------------------
# CREATE SUBPLOTS
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]

    # Filter files for SSP245 and SSP585 only
    file_245 = [f for f in files if model in f and "ssp245" in f]
    file_585 = [f for f in files if model in f and "ssp585" in f]

    if not file_245 or not file_585:
        continue

    file_245 = file_245[0]
    file_585 = file_585[0]

    df_245 = pd.read_csv(file_245)
    df_585 = pd.read_csv(file_585)

    df_245['year_month'] = pd.to_datetime(df_245['year_month'])
    df_585['year_month'] = pd.to_datetime(df_585['year_month'])

    # Filter to non-winter months (March to October only)
    df_245 = df_245[df_245['year_month'].dt.month.between(3, 10)]
    df_585 = df_585[df_585['year_month'].dt.month.between(3, 10)]

    df_245['year'] = df_245['year_month'].dt.year
    df_585['year'] = df_585['year_month'].dt.year

    annual_245 = df_245.groupby('year')['Predicted_TOC'].mean().reset_index()
    annual_585 = df_585.groupby('year')['Predicted_TOC'].mean().reset_index()

    # Mann-Kendall tests for SSP245 and SSP585
    trend_245, p_value_245, z_score_245, slope_245, _ = mann_kendall_test(
        annual_245['year'], annual_245['Predicted_TOC']
    )
    trend_585, p_value_585, z_score_585, slope_585, _ = mann_kendall_test(
        annual_585['year'], annual_585['Predicted_TOC']
    )

    results.append([
        model, 'SSP245', trend_245, p_value_245, z_score_245, slope_245,
        "Significant" if (p_value_245 and p_value_245 < 0.05) else "Not Significant"
    ])
    results.append([
        model, 'SSP585', trend_585, p_value_585, z_score_585, slope_585,
        "Significant" if (p_value_585 and p_value_585 < 0.05) else "Not Significant"
    ])

 # --- PLOT THE FAINTER LINES, now with higher alpha & linewidth ---
    ax.plot(
        annual_245['year'], annual_245['Predicted_TOC'],
        color='blue', alpha=0.15, linewidth=3.5, label="SSP245"
    )
    ax.plot(
        annual_585['year'], annual_585['Predicted_TOC'],
        color='red',  alpha=0.15, linewidth=3.5, label="SSP585"
    )

    # Trend lines based on Sen's slope, now thicker
    line_245 = mann_kendall_sen_slope(annual_245['year'], annual_245['Predicted_TOC'])
    line_585 = mann_kendall_sen_slope(annual_585['year'], annual_585['Predicted_TOC'])

    if line_245 is not None:
        ax.plot(
            annual_245['year'], line_245,
            color='darkblue', linewidth=3.5, label="SSP245 Trend"
        )
    if line_585 is not None:
        ax.plot(
            annual_585['year'], line_585,
            color='darkred', linewidth=3.5, label="SSP585 Trend"
        )

    y_min, y_max = ax.get_ylim()
    y_offset = 0.03 * (y_max - y_min)
    
    
    # Set x-axis ticks to 2040, 2060, 2080, 2100
    ax.set_xticks([2040, 2060, 2080, 2100])
    ax.set_xticklabels([2040, 2060, 2080, 2100])

    # # Set y-axis ticks  (SACHEON RESERVOIR)
    ax.set_yticks([2.0, 2.5, 3.0, 3.5])
    ax.set_yticklabels([2.0, 2.5, 3.0, 3.5])
    
    # Set y-axis ticks  (DOCHEON RESERVOIR)
    # ax.set_yticks([3.0, 3.5, 4.0, 4.5])
    # ax.set_yticklabels([3.0, 3.5, 4.0, 4.5])

    # Clean up axis labels for a neater layout
    if i % 3 != 0:
        ax.set_ylabel('')
        ax.yaxis.set_visible(False)
    if i < 9:
        ax.set_xlabel('')
        ax.xaxis.set_visible(False)

    # Display model name at the top center of each subplot
    ax.text(0.5, 0.85, model, ha='center', va='center', fontsize=28, fontweight='bold', transform=ax.transAxes)

    # Set y-axis limits and adjust tick label sizes
    ax.set_ylim(2, 3.7) # (SACHEON RESERVOIR)
    # ax.set_ylim(3, 4.7) # (DOCHEON RESERVOIR)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

       # Bold the tick labels for both axes
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

# Remove the y-axis and x-axis labels
fig.text(0.05, 0.5, '', va='center', rotation='vertical', fontsize=34, fontweight='bold')
fig.text(0.5, 0.05, '', ha='center', va='center', fontsize=34, fontweight='bold')

# Legend (center top)
handles = [
    plt.Line2D([0], [0], color='darkblue', linewidth=6, label='SSP2-4.5'),
    plt.Line2D([0], [0], color='darkred', linewidth=6, label='SSP5-8.5')
]

# Global Y-axis label
fig.text(0.05, 0.5, 'TOC (mg/L)', va='center', rotation='vertical', fontsize=40, fontweight='bold')

# Place the legend at the top center without overlapping the figure
fig.legend(handles=handles, loc='upper center', fontsize=40, frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.04))

plt.subplots_adjust(left=0.12, bottom=0.12, top=0.93, right=0.98, hspace=0.15, wspace=0.05)
plt.show()

# Build results DataFrame
results_df = pd.DataFrame(results, columns=[
    'Model', 'Scenario', 'Trend', 'P-Value', 'Z-Score', "Sen's Slope", 'Significance'
])

# # ─── ADDITION TO SAVE CSV ───────────────────────────────────────────────────
# # Set this to wherever you want the CSV written:

output_file = r"C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\1-사천\3-PREDICTIONS\11111-TESTING\RESULTSS-DOCHEON\Boosted-M5\RESULTS\MK_SenSlope_Results(DOCHEON).csv"
results_df.to_csv(output_file, index=False)
print(f"Mann-Kendall results saved to: {output_file}")