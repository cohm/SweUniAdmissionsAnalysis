import pandas as pd
import numpy as np
import re
import swemaps
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

def load_excel_files(file_paths):
    """Load and combine multiple Excel files with student data."""
    all_data = pd.concat([pd.read_excel(fp) for fp in file_paths], ignore_index=True)
    print("Read in data for %d applicants" % len(all_data))
    #print(all_data.dtypes)
    return all_data

def load_postal_mapping(mapping_csv):
    """Load postal code to municipality mapping."""
    mapping = pd.read_csv(mapping_csv, dtype={'Postnummer': str})
    mapping['Postnummer'] = mapping['Postnummer'].str.zfill(5)  # Ensure 5-digit codes
    print("Read in data for %d postal codes" % len(mapping))
    return mapping

def preprocess_data(df, mapping, postal_code_col):
    """Pre-process the data, and add info such as county names from addresses (via postal code mapping data), extract merit scores into separate columns, etc"""
    
    # start with geographic data
    df = df.copy()
    df[postal_code_col] = df[postal_code_col].astype(str).str.zfill(5)
    df = df.merge(mapping, how='left', left_on=postal_code_col, right_on='Postnummer')
    df = df.dropna(subset=['LnNamn'])
    df = df.astype({'KnKod':'Int64'}) # this was otherwise of type 'object'

    # now extract the grade data
    def extract_score(merit, pattern):
        if pd.isna(merit):
            return None
        match = pattern.search(merit)
        if match:
            return float(match.group(1).replace(",", "."))
        return None

    # add the merit score cols
    merit_types = [ "BI", "BII", "HP", "MAFY" ]
    for merit_type in merit_types:
        merit_pattern = re.compile(rf"{merit_type}\s*\(\s*([\d.,]+)\s*\)", flags=re.IGNORECASE)
        df = df.copy()
        df[f"{merit_type}_score"] = df["Meritvärde"].apply(lambda x: extract_score(x, merit_pattern))

    # and a column for whether the applicant was accepted (NB! The same applicant can have several rows if they applied to more than one program)
    df['Antagen'] = df['Resultat'].str.contains("Antagen")

    # now extract the gender via the personal number
    def extract_gender(personal_number):
        if int(str(personal_number)[11]) % 2 == 0:
            return "K"
        else:
            return "M"
    df['Kön'] = df['Personnummer'].apply(extract_gender)

    print(df)
    return df

def aggregate_by_column(df, col_name):
    """Count number of rows in df that have the same value for column with name col_name, e.g. to get the number of applicants from each municipality."""
    return df[col_name].value_counts().reset_index(name='number')

def plot_municipality_map(aggregated_df, group = "", printTotal = True):
    # load GeoParquet data via swemaps.get_path("kommun")
    path = swemaps.get_path("kommun")
    map_df = gpd.read_parquet(path)
    map_df = map_df.astype({'kommun_kod':'Int64'})

    print(map_df)
    print(aggregated_df)
    # columns 'kommun_kod' and 'KnKod' should be matched to merge
    merged = map_df.merge(aggregated_df, left_on="kommun_kod", right_on="KnKod", how="left")
    merged["number"] = merged["number"].fillna(0)
    print(merged)

    fig, ax = plt.subplots(1, 1, figsize=(7, 12))
    merged.plot(
        column="number",
        cmap="Blues",
        linewidth=0.8,
        edgecolor="0.8",
        ax=ax,
        legend=True,
        norm=LogNorm(vmin=1, vmax=merged["number"].max()),
        missing_kwds={"color": "lightgrey", "label": "No data"}
    )
    
    # add labels with number of applicants for each region
    total = 0 # keep track of total, and draw this at the end
    for _, row in merged.iterrows():
        if row["number"] > 0:
            total += row["number"]
            centroid = row["geometry"].centroid
            ax.text(
                centroid.x, centroid.y,
                f"{int(row['number'])}",
                ha="center", va="center",
                fontsize=6,
                color="black",
                alpha=0.7
            )
    if printTotal:
        print(group + ": %d" % total)

    ax.set_title(group + " per kommun" + (f" (totalt {int(total)})" if printTotal else ""), fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(("plots/MunicipalityMap_%s.pdf" % group).replace(' ', '_'), dpi=300)
    plt.show()
    
def plot_county_map(aggregated_df, group = "", printTotal = True):
    # load GeoParquet data via swemaps.get_path("lan")
    path = swemaps.get_path("lan")
    map_df = gpd.read_parquet(path)
    # columns 'lan' and 'LnNamn' should be matched to merge
    merged = map_df.merge(aggregated_df, left_on="lan", right_on="LnNamn", how="left")
    merged["number"] = merged["number"].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 12))
    merged.plot(
        column="number",
        cmap="Blues",
        linewidth=0.8,
        edgecolor="0.8",
        ax=ax,
        legend=True,
        norm=LogNorm(vmin=1, vmax=merged["number"].max()),
        missing_kwds={"color": "lightgrey", "label": "No data"}
    )
    
    # add labels with number of applicants for each region
    total = 0 # keep track of total, and draw this at the end
    for _, row in merged.iterrows():
        if row["number"] > 0:
            total += row["number"]
            centroid = row["geometry"].centroid
            ax.text(
                centroid.x, centroid.y, f"{int(row['number'])}", ha="center", va="center",
                fontsize=6, color="black", alpha=0.7
            )
    if printTotal:
        print(group + ": %d" % total)
    
    ax.set_title(group + " per län" + (f" (totalt {int(total)})" if printTotal else ""), fontsize=16)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(("plots/CountyMap_%s.pdf" % group).replace(' ', '_'), dpi=300)
    plt.show()
    
def plot_score_distribution(df, category, group):
    """
    Plot histogram of a score category ('BI', 'BII', 'HP' or 'MAFY') for all applicants and admitted ones. Bin sizes depends on category. Adds counts to legend.
    """

    bin_sizes = {
        "BI": 0.1,
        "BII": 0.1,
        "HP": 0.05,
        "MAFY": 1
    }

    if category not in bin_sizes:
        raise ValueError(f"Unknown category '{category}'. Must be one of: {list(bin_sizes.keys())}")

    bin_size = bin_sizes[category]

    pattern = re.compile(rf"{category}\s*\(\s*([\d.,]+)\s*\)", flags=re.IGNORECASE)

    def extract_score(merit):
        if pd.isna(merit):
            return None
        match = pattern.search(merit)
        if match:
            return float(match.group(1).replace(",", "."))
        return None

    df = df.copy()
    df[f"{category}_score"] = df["Meritvärde"].apply(extract_score)

    has_score = df[df[f"{category}_score"].notna()]
    admitted = has_score[has_score["Resultat"].str.lower() == "antagen"]

    if has_score.empty:
        print(f"No valid {category} scores found.")
        return

    total_all = len(has_score)
    total_admitted = len(admitted)

    min_val = has_score[f"{category}_score"].min()
    max_val = has_score[f"{category}_score"].max()
    bins = np.arange(np.floor(min_val)-bin_size/2, np.ceil(max_val)+bin_size/2, bin_size)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)  # Grid behind bars

    ax.hist(
        has_score[f"{category}_score"],
        bins=bins,
        alpha=0.5,
        label=f"Alla sökande (n = {total_all})",
        color="gray",
        edgecolor="black"
    )
    ax.hist(
        admitted[f"{category}_score"],
        bins=bins,
        alpha=0.5,
        label=f"Antagna (n = {total_admitted})",
        color="blue",
        edgecolor="black"
    )

    ax.set_xlabel(f"{category}-poäng")
    ax.set_ylabel("Antal sökande")
    ax.set_title(f"Fördelning av {category}-poäng, {group}: alla sökande vs. antagna")

    # Subtle grid
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.3)

    ax.legend()
    plt.tight_layout()
    plt.savefig(("plots/GradeDistribution_%s_%s.pdf" % (group, category)).replace(' ', '_'), dpi=300)
    plt.show()

def plot_merit_correlation(df, group, merit1, merit2):
    """
    Plot correlation between merit1 and merit2 scores.
    Draws three regression lines (all, admitted, rejected),
    shows Pearson and Spearman correlations,
    and includes a count-based legend.
    """

    valid = df.dropna(subset=[f"{merit1}_score", f"{merit2}_score"])

    if valid.empty:
        print("No valid data points with both {merit1} and {merit2} scores.")
        return

    # Subsets
    admitted = valid[valid["Antagen"]]
    rejected = valid[~valid["Antagen"]]

    # Counts
    total_all = len(valid)
    total_admitted = len(admitted)
    total_rejected = len(rejected)

    # Correlations (for all data)
    pearson_corr, _ = pearsonr(valid[f"{merit1}_score"], valid[f"{merit2}_score"])
    spearman_corr, _ = spearmanr(valid[f"{merit1}_score"], valid[f"{merit2}_score"])

    # Plot
    plt.figure(figsize=(9, 7))
    sns.set(style="whitegrid")

    # All candidates - solid black regression line
    sns.regplot(
        data=valid,
        x=f"{merit1}_score",
        y=f"{merit2}_score",
        scatter=False,
        line_kws={"color": "black", "label": f"All (n={total_all})", "linestyle": "-"},
        ci=None
    )

    # Admitted - dashed blue
    if not admitted.empty:
        sns.regplot(
            data=admitted,
            x=f"{merit1}_score",
            y=f"{merit2}_score",
            scatter=False,
            line_kws={"color": "blue", "label": f"Antagna (n = {total_admitted})", "linestyle": "--"},
            ci=None
        )

    # Rejected - dashed gray
    if not rejected.empty:
        sns.regplot(
            data=rejected,
            x=f"{merit1}_score",
            y=f"{merit2}_score",
            scatter=False,
            line_kws={"color": "gray", "label": f"Ej antagna (n = {total_rejected})", "linestyle": "--"},
            ci=None
        )

    # Scatter plot (colored by admission)
    sns.scatterplot(
        data=valid,
        x=f"{merit1}_score",
        y=f"{merit2}_score",
        hue="Antagen",
        palette={True: "blue", False: "gray"},
        edgecolor="black",
        alpha=0.6
    )

    plt.title(
        f"Korrelation mellan {merit1}- och {merit2}-poäng, sökande {group}\n"
        f"(Pearson r = {pearson_corr:.3f}, Spearman ρ = {spearman_corr:.3f})"
    )
    plt.xlabel(f"{merit1}-poäng")
    plt.ylabel(f"{merit2}-poäng")
    plt.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
    plt.legend(title="Antagen")
    plt.tight_layout()
    plt.savefig(("plots/GradeCorrelation_%s_%s-%s.pdf" % (group, merit1, merit2)).replace(' ', '_'), dpi=300)
    plt.show()

if __name__ == "__main__":
    # Update with the actual paths to your Excel files
    excel_files = [
        '../../NyA/createReport-2025.xls',
        # Add more paths if needed
    ]
    postalcode_mapping_file = 'data/postalcode_data.csv'
    postalcode_column_name = 'Postnummer'

    df = load_excel_files(excel_files)
    mapping = load_postal_mapping(postalcode_mapping_file)
    #print(mapping)
    df = preprocess_data(df, mapping, postalcode_column_name)
    print(df[0:10])
    aggMunicipality = aggregate_by_column(df, 'KnKod')
    print("Will now print the aggregated counts per municipality")
    print(aggMunicipality)
    plot_municipality_map(aggMunicipality, "CING SCI sökande")
    aggCounty = aggregate_by_column(df, 'LnNamn')
    print(aggCounty)
    plot_county_map(aggCounty, "CING SCI sökande")
    
    programs = ["CTFYS"] #, "CTMAT", "CFATE", "COPEN"]
    for p in programs:
        print("Will now check program %s specifically" % p)
        df_program = df[df['Kurs-/programkod'].str.contains(p)]
        df_program_prio = df_program[df_program['Prio'] == 1]
        df_program_admitted = df_program[df_program['Resultat'].str.contains('Antagen')]
        #df_program_admitted_accepted = df_program_admitted[df_program_admitted['Ja-svar'].str.contains('Ja')]
        print(df_program)
        #print(df_program)
        aggMunicipality = aggregate_by_column(df_program, 'KnKod')
        aggMunicipalityPrio = aggregate_by_column(df_program_prio, 'KnKod')
        aggMunicipalityAdmitted = aggregate_by_column(df_program_admitted, 'KnKod')
        print("Will now print the aggregated counts per municipality for %s" % p)
        print(aggMunicipality)
        print(aggMunicipalityAdmitted)
        plot_municipality_map(aggMunicipality, p+" sökande")
        plot_municipality_map(aggMunicipalityPrio, p+" sökande prio 1")
        plot_municipality_map(aggMunicipalityAdmitted, p+" antagna")
        aggCounty = aggregate_by_column(df_program, 'LnNamn')
        aggCountyPrio = aggregate_by_column(df_program_prio, 'LnNamn')
        aggCountyAdmitted = aggregate_by_column(df_program_admitted, 'LnNamn')
        #aggCountyAdmittedAccepted = aggregate_by_column(df_program_admitted_accepted, 'LnNamn')
        print("Will now print the aggregated counts per county for %s" % p)
        print(aggCounty)
        print(aggCountyAdmitted)
        #print(aggCountyAdmittedAccepted)
        plot_county_map(aggCounty, p+" sökande")
        plot_county_map(aggCountyPrio, p+" sökande prio 1")
        plot_county_map(aggCountyAdmitted, p+" antagna")
        #plot_county_map(aggCountyAdmittedAccepted, p+" tackat ja")

        # test with grade distributions too
        plot_score_distribution(df_program, "BI", p)
        plot_score_distribution(df_program, "BII", p)
        plot_score_distribution(df_program, "HP", p)
        plot_score_distribution(df_program, "MAFY", p)

        # look at some correlations between grade categories
        plot_merit_correlation(df_program, p, "HP", "BI")
        plot_merit_correlation(df_program, p, "MAFY", "BI")
        plot_merit_correlation(df_program, p, "BII", "BI")
        plot_merit_correlation(df_program, p, "HP", "MAFY")
