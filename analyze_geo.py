import pandas as pd
import numpy as np
import swemaps
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def load_excel_files(file_paths):
    """Load and combine multiple Excel files with student data."""
    all_data = pd.concat([pd.read_excel(fp) for fp in file_paths], ignore_index=True)
    print("Read in data for %d applicants" % len(all_data))
    print(all_data.dtypes)
    return all_data

def load_postal_mapping(mapping_csv):
    """Load postal code to municipality mapping."""
    mapping = pd.read_csv(mapping_csv, dtype={'Postnummer': str})
    mapping['Postnummer'] = mapping['Postnummer'].str.zfill(5)  # Ensure 5-digit codes
    print("Read in data for %d postal codes" % len(mapping))
    return mapping

def preprocess_data(df, mapping, postal_code_col):
    """Add municipality and county names from addresses, via postal code mapping data."""
    df = df.copy()
    df[postal_code_col] = df[postal_code_col].astype(str).str.zfill(5)
    df = df.merge(mapping, how='left', left_on=postal_code_col, right_on='Postnummer')
    df = df.dropna(subset=['LnNamn'])
    df = df.astype({'KnKod':'Int64'}) # this was otherwise of type 'object'
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
    
    programs = ["CTFYS", "CTMAT", "CFATE", "COPEN"]
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
        print("Dataframe")
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

