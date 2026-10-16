import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def _get_usable_spectral_cols(df):
    """Identifies usable spectral bands (drops 100% NaN bands) and sorts them by wavelength[cite: 2]."""
    spec_cols = [c for c in df.columns if c.startswith('X')]
    na_frac = df[spec_cols].isna().mean()
    usable_cols = na_frac[na_frac < 1.0].index.tolist()
    
    # Extract integer wavelengths and sort
    wavelengths = [int(c[1:]) for c in usable_cols]
    order = np.argsort(wavelengths)
    
    usable_sorted = np.array(usable_cols)[order]
    wavelengths_sorted = np.array(wavelengths)[order]
    return usable_sorted, wavelengths_sorted

def _ensure_spectral_sum(df, usable_cols):
    """Calculates spectral_sum if it does not already exist[cite: 2]."""
    if 'spectral_sum' not in df.columns:
        df['spectral_sum'] = df[usable_cols].sum(axis=1)
    return df


# --------- GRAPH --------


def plot_crop_distribution(df, output_dir):
    """Basic bar chart of the crop target class distribution."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Crop', order=df['Crop'].value_counts().index, palette='colorblind')
    plt.title('Crop Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_crop_distribution.png'))
    plt.close()


def plot_spectral_sum_boxplot(df, output_dir):
    """Boxplot displaying the variance in overall spectral sum across crops[cite: 2]."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Crop', y='spectral_sum', palette='colorblind')
    plt.title('Spectral Sum vs Crop')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_spectral_sum_boxplot.png'))
    plt.close()


def plot_crop_stage_heatmap(df, output_dir):
    """Heatmap showing the count of instances for each Crop and Stage combination[cite: 2]."""
    plt.figure(figsize=(10, 6))
    crosstab = pd.crosstab(df['Crop'], df['Stage'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('Crop vs. Stage Distribution Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_crop_stage_heatmap.png'))
    plt.close()

def plot_mean_spectral_signature(df, usable_cols, wavelengths, output_dir):
    """Line plot showing the average reflectance per wavelength for each crop[cite: 2]."""
    plt.figure(figsize=(10, 5))
    for crop, group in df.groupby('Crop'):
        mean_signature = group[usable_cols].mean()
        plt.plot(wavelengths, mean_signature.values, label=crop)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Mean Reflectance')
    plt.title('Mean Spectral Signature per Crop')
    plt.legend(title='Crop')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_mean_spectral_signature.png'))
    plt.close()

def plot_spectral_variance(df, usable_cols, wavelengths, output_dir):
    """Visualizes the standard deviation bands around the mean spectral signature."""
    plt.figure(figsize=(12, 6))
    for crop, group in df.groupby('Crop'):
        mean_sig = group[usable_cols].mean().values
        std_sig = group[usable_cols].std().values
        
        p = plt.plot(wavelengths, mean_sig, label=crop)
        color = p[0].get_color()
        plt.fill_between(wavelengths, mean_sig - std_sig, mean_sig + std_sig, color=color, alpha=0.1)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Spectral Signature Variance (Mean +/- 1 STD) per Crop')
    plt.legend(title='Crop')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_spectral_variance.png'))
    plt.close()


def plot_band_correlation(df, usable_cols, output_dir):
    """Correlation matrix of sampled spectral bands to highlight collinearity[cite: 3]."""
    # Sample every 5th band to keep the heatmap legible
    sampled_bands = usable_cols[::5]
    corr = df[sampled_bands].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False)
    plt.title('Spectral Band Correlation Matrix (Sampled)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_band_correlation_matrix.png'))
    plt.close()


def plot_pca_2d_projection(df, usable_cols, output_dir):
    """Projects the spectral data into 2D using PCA to observe crop separability[cite: 3]."""
    # Impute missing values and scale data before PCA
    X_imputed = SimpleImputer(strategy='median').fit_transform(df[usable_cols])
    X_scaled = StandardScaler().fit_transform(X_imputed)
    
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Crop'] = df['Crop'].values
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Crop', palette='colorblind', alpha=0.7)
    plt.title(f'PCA 2D Projection (Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_pca_2d_projection.png'))
    plt.close()


def plot_stage_evolution_per_crop(df, usable_cols, wavelengths, output_dir):
    """Small multiples showing the mean spectral signature across Growth Stages for each crop[cite: 2]."""
    crops = df['Crop'].unique()
    num_crops = len(crops)
    
    fig, axes = plt.subplots(num_crops, 1, figsize=(10, 4 * num_crops), sharex=True)
    if num_crops == 1:
        axes = [axes]
        
    for ax, crop in zip(axes, crops):
        crop_data = df[df['Crop'] == crop]
        for stage, group in crop_data.groupby('Stage'):
            mean_sig = group[usable_cols].mean()
            ax.plot(wavelengths, mean_sig.values, label=stage)
            
        ax.set_title(f'Stage Evolution: {crop.capitalize()}')
        ax.set_ylabel('Mean Reflectance')
        ax.grid(alpha=0.3)
        ax.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_stage_evolution_per_crop.png'))
    plt.close()


def generate_all_eda(data_path, output_dir):
    """Loads data, prepares prerequisites, and generates all EDA artifacts."""
    print("Loading data for EDA...")
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prep features
    usable_cols, wavelengths = _get_usable_spectral_cols(df)
    df = _ensure_spectral_sum(df, usable_cols)
    
    print("Generating Graph 1: Crop Distribution...")
    plot_crop_distribution(df, output_dir)
    
    print("Generating Graph 2: Spectral Sum Boxplot...")
    plot_spectral_sum_boxplot(df, output_dir)
    
    print("Generating Graph 3: Crop/Stage Heatmap...")
    plot_crop_stage_heatmap(df, output_dir)
    
    print("Generating Graph 4: Mean Spectral Signature...")
    plot_mean_spectral_signature(df, usable_cols, wavelengths, output_dir)
    
    print("Generating Graph 5: Spectral Variance Plot...")
    plot_spectral_variance(df, usable_cols, wavelengths, output_dir)
    
    print("Generating Graph 6: Band Correlation Matrix...")
    plot_band_correlation(df, usable_cols, output_dir)
    
    print("Generating Graph 7: PCA 2D Projection...")
    plot_pca_2d_projection(df, usable_cols, output_dir)
    
    print("Generating Graph 8: Stage Evolution per Crop...")
    plot_stage_evolution_per_crop(df, usable_cols, wavelengths, output_dir)
    
    print("EDA Generation Complete.")

# if __name__ == "__main__":
#     # Test script execution
#     generate_all_eda('data/train.csv', 'output/eda_plots/')