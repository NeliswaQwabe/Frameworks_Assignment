#PART 1: CORD-19 Dataset Analysis
# Step 1: Download and Load the Data
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import numpy as np

# Load the dataset with error handling
try:
    if not os.path.exists("metadata.csv"):
        raise FileNotFoundError("metadata.csv not found in the current directory")
    
    df = pd.read_csv("metadata.csv", low_memory=False)
    print("✅ File loaded successfully!\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Examine the first few rows
print("📊 First 5 rows of the dataset:")
print(df.head())

# Check the data structure
print("\n📈 Dataset shape (rows, columns):", df.shape)
print("\nℹ️ DataFrame info:")
df.info()

# Check for missing data
print("\n🚨 Missing values in the first 10 columns:")
print(df.isnull().sum().head(10))

# Step 2: Basic Data Exploration
print("\n=== Basic Data Exploration ===")

# Check the DataFrame dimensions (rows, columns)
print("\n➡️ Dataset dimensions:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Identify data types of each column
print("\n➡️ Data types of each column:")
print(df.dtypes)

# Check for missing values in important columns
important_cols = ["title", "abstract", "authors", "journal", "publish_time"]
existing_cols = [col for col in important_cols if col in df.columns]
print("\n➡️ Missing values in important columns:")
print(df[existing_cols].isnull().sum())

# Generate basic statistics for numerical columns
print("\n➡️ Basic statistics for numerical columns:")
print(df.describe())

# PART 2: Data Cleaning and Preparation
print("\n\n=== PART 2: Data Cleaning and Preparation ===")

# Step 1: Handle Missing Data
print("\n📋 Step 1: Identify columns with missing values")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': missing_data,
    'Percentage': missing_percent
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Percentage', ascending=False)
print(missing_summary)

# Step 2: Decision on handling missing values
print("\n🔧 Step 2: Handle missing values")
df_cleaned = df.copy()

# Fill missing abstract with empty string
if 'abstract' in df_cleaned.columns:
    df_cleaned['abstract'].fillna('', inplace=True)
    print("✅ Filled missing abstracts with empty strings")

# Fill missing authors with 'Unknown'
if 'authors' in df_cleaned.columns:
    df_cleaned['authors'].fillna('Unknown', inplace=True)
    print("✅ Filled missing authors with 'Unknown'")

# Fill missing journal with 'Unknown'
if 'journal' in df_cleaned.columns:
    df_cleaned['journal'].fillna('Unknown', inplace=True)
    print("✅ Filled missing journal with 'Unknown'")

# Remove rows with missing title (critical column)
if 'title' in df_cleaned.columns:
    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['title'].notna()]
    rows_removed = rows_before - len(df_cleaned)
    print(f"✅ Removed {rows_removed} rows with missing titles")

# Step 3: Prepare data for analysis
print("\n⏰ Step 3: Convert date columns and extract year")

# Convert publish_time to datetime
if 'publish_time' in df_cleaned.columns:
    df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
    print("✅ Converted publish_time to datetime format")
    
    # Extract year from publication date
    df_cleaned['publication_year'] = df_cleaned['publish_time'].dt.year
    print("✅ Extracted publication year")

# Step 4: Create new columns
print("\n✨ Step 4: Create new feature columns")

# Calculate abstract word count
if 'abstract' in df_cleaned.columns:
    df_cleaned['abstract_word_count'] = df_cleaned['abstract'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    print("✅ Created abstract_word_count column")

# Calculate title word count
if 'title' in df_cleaned.columns:
    df_cleaned['title_word_count'] = df_cleaned['title'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    print("✅ Created title_word_count column")

# Step 5: Display cleaned data summary
print("\n📊 Step 5: Cleaned dataset summary")
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")
print(f"\n✅ Remaining missing values:\n{df_cleaned.isnull().sum().sum()} total missing values")

print("\n📈 Sample of cleaned data:")
print(df_cleaned[['title', 'abstract_word_count', 'publication_year', 'journal']].head(10))

print("\n🎯 Data cleaning and preparation completed successfully!")

# PART 3: Data Analysis and Visualization
print("\n\n=== PART 3: Data Analysis and Visualization ===")

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Step 1: Count papers by publication year
print("\n📅 Step 1: Analyze papers by publication year")
if 'publication_year' in df_cleaned.columns:
    papers_by_year = df_cleaned['publication_year'].value_counts().sort_index()
    print(f"\n📊 Papers by year:\n{papers_by_year}")
    print(f"Year range: {papers_by_year.index.min()} - {papers_by_year.index.max()}")

# Step 2: Identify top journals
print("\n🏆 Step 2: Identify top journals publishing COVID-19 research")
if 'journal' in df_cleaned.columns:
    top_journals = df_cleaned['journal'].value_counts().head(15)
    print(f"\n📚 Top 15 journals:\n{top_journals}")

# Step 3: Find most frequent words in titles
print("\n📝 Step 3: Find most frequent words in paper titles")
if 'title' in df_cleaned.columns:
    # Combine all titles and convert to lowercase
    all_titles = ' '.join(df_cleaned['title'].astype(str)).lower()
    
    # Split into words and filter common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                  'covid', '19', 'coronavirus', 'sars', 'cov', 'covid-19', 'novel',
                  'new', 'study', 'research', 'analysis', 'case', 'effect', 'related'}
    
    words = [word for word in all_titles.split() if word.isalpha() and len(word) > 3 and word not in stop_words]
    word_freq = Counter(words)
    top_words = word_freq.most_common(20)
    
    print(f"\n🔤 Top 20 most frequent words in titles:")
    for word, count in top_words:
        print(f"   {word}: {count}")

# Step 4: Create visualizations
print("\n🎨 Step 4: Creating visualizations...")

# Visualization 1: Publications over time
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Line plot of publications over time
if 'publication_year' in df_cleaned.columns:
    papers_by_year_sorted = df_cleaned['publication_year'].value_counts().sort_index()
    axes[0, 0].plot(papers_by_year_sorted.index, papers_by_year_sorted.values, marker='o', linewidth=2, color='#2E86AB')
    axes[0, 0].set_title('Number of COVID-19 Publications Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Publication Year', fontsize=12)
    axes[0, 0].set_ylabel('Number of Papers', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    print("✅ Created publications over time plot")

# Plot 2: Bar chart of top publishing journals
if 'journal' in df_cleaned.columns:
    top_journals_plot = df_cleaned['journal'].value_counts().head(15)
    axes[0, 1].barh(range(len(top_journals_plot)), top_journals_plot.values, color='#A23B72')
    axes[0, 1].set_yticks(range(len(top_journals_plot)))
    axes[0, 1].set_yticklabels(top_journals_plot.index, fontsize=10)
    axes[0, 1].set_title('Top 15 Journals Publishing COVID-19 Research', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Papers', fontsize=12)
    axes[0, 1].invert_yaxis()
    print("✅ Created top journals bar chart")

# Plot 3: Bar chart of top words in titles
if 'title' in df_cleaned.columns:
    words_list, counts_list = zip(*top_words)
    axes[1, 0].bar(range(len(top_words)), counts_list, color='#F18F01')
    axes[1, 0].set_xticks(range(len(top_words)))
    axes[1, 0].set_xticklabels(words_list, rotation=45, ha='right', fontsize=10)
    axes[1, 0].set_title('Top 20 Most Frequent Words in Paper Titles', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    print("✅ Created word frequency chart")

# Plot 4: Distribution of papers by source
if 'source_x' in df_cleaned.columns or 'source' in df_cleaned.columns:
    source_col = 'source_x' if 'source_x' in df_cleaned.columns else 'source'
    top_sources = df_cleaned[source_col].value_counts().head(10)
    axes[1, 1].bar(range(len(top_sources)), top_sources.values, color='#06A77D')
    axes[1, 1].set_xticks(range(len(top_sources)))
    axes[1, 1].set_xticklabels(top_sources.index, rotation=45, ha='right', fontsize=10)
    axes[1, 1].set_title('Distribution of Paper Counts by Source', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Papers', fontsize=12)
    print("✅ Created source distribution chart")
else:
    # Alternative: Show distribution if source column doesn't exist
    axes[1, 1].text(0.5, 0.5, 'Source column not available in dataset', 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('Distribution of Paper Counts by Source', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('covid19_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization as 'covid19_analysis_visualizations.png'")
plt.show()

# Create Word Cloud
print("\n☁️ Step 5: Creating word cloud...")
if 'title' in df_cleaned.columns:
    all_titles_for_wordcloud = ' '.join(df_cleaned['title'].astype(str))
    
    wordcloud = WordCloud(width=1200, height=600, background_color='white', 
                          colormap='viridis', max_words=100).generate(all_titles_for_wordcloud)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of COVID-19 Paper Titles', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('covid19_wordcloud.png', dpi=300, bbox_inches='tight')
    print("✅ Saved word cloud as 'covid19_wordcloud.png'")
    plt.show()

print("\n🎉 Data analysis and visualization completed successfully!")
print(f"\n📁 Files saved:")
print("   - covid19_analysis_visualizations.png")
print("   - covid19_wordcloud.png")