import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

st.set_page_config(layout="wide")
st.title("Field Data Analysis")

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))

# -------------------------
# SESSION
# -------------------------

if "control" not in st.session_state:
    st.session_state.control = set()

if "groups" not in st.session_state:
    st.session_state.groups = {}

if "excluded_sites" not in st.session_state:
    st.session_state.excluded_sites = {}

# -------------------------
# SIDEBAR
# -------------------------

if st.sidebar.button("Reset Experiment"):
    st.session_state.control = set()
    st.session_state.groups = {}
    st.session_state.excluded_sites = {}
    st.rerun()

file = st.sidebar.file_uploader("Upload Field CSV")

metric = st.sidebar.selectbox("Metric", [
    "MAP2_area",
    "NeuN_count",
    "pSyn_over_MAP2",
    "pSyn_over_NeuN"
])

group_name = st.sidebar.text_input("Active group", "Group1")

# -------------------------
# HELPERS
# -------------------------

def clean_well(x):
    m = re.match(r"([A-H])\s*-\s*(\d+)", str(x))
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return x

def prepare(df):
    df["Well"] = df["WELL LABEL"].apply(clean_well)

    df["MAP2_area"] = df["MAP2 area_MAP2_Area_Sum"]
    df["NeuN_count"] = df["NeuN count_NeuN Count"]
    df["pSyn_area"] = df["pSyn Area_pSyn_Area_Sum"]

    df["pSyn_over_MAP2"] = np.where(df["MAP2_area"]>0, df["pSyn_area"]/df["MAP2_area"], 0)
    df["pSyn_over_NeuN"] = np.where(df["NeuN_count"]>0, df["pSyn_area"]/df["NeuN_count"], 0)

    return df

def filter_sites(df):
    keep = []
    for _, row in df.iterrows():
        well = row["Well"]
        fov = int(row["FOV"])
        if well in st.session_state.excluded_sites:
            if fov in st.session_state.excluded_sites[well]:
                keep.append(False)
                continue
        keep.append(True)
    return df[keep]

# -------------------------
# HEATMAPS
# -------------------------

def site_heatmap(df, col, vmin, vmax):

    grid = np.full((8*3, 12*3), np.nan)

    for _, row in df.iterrows():
        well = row["Well"]
        fov = int(row["FOV"]) - 1

        r = ROWS.index(well[0])
        c = int(well[1:]) - 1

        i = r*3 + (fov//3)
        j = c*3 + (fov%3)

        grid[i,j] = row[col]

    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_xticks([i*3+1 for i in range(12)])
    ax.set_xticklabels(COLS)
    ax.set_yticks([i*3+1 for i in range(8)])
    ax.set_yticklabels(ROWS)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    for i in range(0, 8*3, 3):
        ax.axhline(i-0.5, color='white', linewidth=2)
    for j in range(0, 12*3, 3):
        ax.axvline(j-0.5, color='white', linewidth=2)

    for r_idx, r in enumerate(ROWS):
        for c_idx, c in enumerate(COLS):
            ax.text(c_idx*3+1, r_idx*3+1, f"{r}{c}",
                    ha="center", va="center", fontsize=6)

    plt.colorbar(im, ax=ax, fraction=0.025)
    return fig


def well_heatmap(df_full, vmin, vmax):

    plate = pd.DataFrame(
        [{"Well": f"{r}{c}", "Row": r, "Col": c} for r in ROWS for c in COLS]
    )

    plate = plate.merge(df_full, on="Well", how="left")

    mat = plate.pivot(index="Row", columns="Col", values="Norm")

    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(mat.values, vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    for i,r in enumerate(ROWS):
        for j,c in enumerate(COLS):
            ax.text(j,i,f"{r}{c}",ha="center",va="center",fontsize=6)

    # highlight control wells
    for well in st.session_state.control:
        r = ROWS.index(well[0])
        c = int(well[1:]) - 1
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5),1,1,
                                   fill=False, edgecolor='red', linewidth=1.5))

    plt.colorbar(im, ax=ax, fraction=0.025)
    return fig

# -------------------------
# BAR PLOT
# -------------------------

def bar_plot(df):

    summary = df.groupby("Group")["Norm"].agg(["mean","sem","count"]).reset_index()

    fig, ax = plt.subplots(figsize=(2.8,2.2))

    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["sem"], capsize=2, width=0.5)

    for i,g in enumerate(summary["Group"]):
        y = df[df["Group"]==g]["Norm"]
        ax.scatter(np.full(len(y),i)+np.random.normal(0,0.02,len(y)), y, s=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["Group"], rotation=30, ha="right", fontsize=7)

    ax.set_ylabel("Normalized", fontsize=7)
    ax.tick_params(axis='y', labelsize=6)

    plt.tight_layout()

    return fig, summary

# -------------------------
# PRISM TABLE
# -------------------------

def prism_table(df):

    groups = sorted(df["Group"].unique())
    data = {}
    max_len = 0

    for g in groups:
        vals = df[df["Group"] == g]["Norm"].reset_index(drop=True)
        data[g] = vals
        max_len = max(max_len, len(vals))

    for g in groups:
        data[g] = data[g].reindex(range(max_len))

    return pd.DataFrame(data)

# -------------------------
# MAIN
# -------------------------

if file:

    df = pd.read_csv(file)
    df = prepare(df)

    wells = sorted(df["Well"].unique())

    # CONTROL
    st.subheader("1. Control Wells")

    for r in ROWS:
        cols = st.columns(12)
        for i,c in enumerate(COLS):
            well = f"{r}{c}"
            if well not in wells:
                continue

            label = f"🟦 {well}" if well in st.session_state.control else well

            if cols[i].button(label, key=f"ctrl_{well}"):
                if well in st.session_state.control:
                    st.session_state.control.remove(well)
                else:
                    st.session_state.control.add(well)
                st.rerun()

    if len(st.session_state.control) == 0:
        st.stop()

    # GROUPS
    st.subheader("2. Assign Groups")

    for r in ROWS:
        cols = st.columns(12)
        for i,c in enumerate(COLS):
            well = f"{r}{c}"
            if well not in wells:
                continue

            label = f"🟩 {well}" if well in st.session_state.groups else well

            if cols[i].button(label, key=f"group_{well}"):
                if well in st.session_state.groups:
                    del st.session_state.groups[well]
                else:
                    st.session_state.groups[well] = group_name
                st.rerun()

    # FILTER
    df_filtered = filter_sites(df)

    # 🔥 FULL DATA (heatmap)
    df_well_full = df_filtered.groupby("Well")[metric].mean().reset_index()

    control_mean = df_well_full[
        df_well_full["Well"].isin(st.session_state.control)
    ][metric].mean()

    df_well_full["Norm"] = df_well_full[metric] / control_mean

    # 🔥 GROUPED DATA (bar plot only)
    df_well = df_well_full.copy()
    df_well["Group"] = df_well["Well"].map(st.session_state.groups)
    df_well.loc[df_well["Well"].isin(st.session_state.control), "Group"] = "Control"
    df_well = df_well.dropna(subset=["Group"])

    df_filtered["Norm"] = df_filtered[metric] / control_mean

    # GLOBAL SCALE
    vmin = np.nanpercentile(df_filtered["Norm"], 5)
    vmax = np.nanpercentile(df_filtered["Norm"], 95)

    # RESULTS
    st.subheader("Results")

    col1, col2 = st.columns([2,2])

    with col1:
        st.pyplot(site_heatmap(df_filtered, "Norm", vmin, vmax))

    with col2:
        st.pyplot(well_heatmap(df_well_full, vmin, vmax))  # ✅ ALWAYS FULL

    fig, summary = bar_plot(df_well)
    st.pyplot(fig)

    st.subheader("Summary")
    st.dataframe(summary)

    st.subheader("Prism Table")
    st.dataframe(prism_table(df_well))