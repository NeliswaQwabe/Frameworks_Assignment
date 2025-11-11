import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import numpy as np

# Set page configuration
st.set_page_config(page_title="CORD-19 Data Explorer", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("🔬 CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers from the CORD-19 dataset")
st.markdown("---")

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("metadata.csv", low_memory=False)
        return df
    except FileNotFoundError:
        st.error("❌ metadata.csv file not found!")
        return None

# Clean and prepare data
@st.cache_data
def prepare_data(df):
    df_cleaned = df.copy()
    
    # Fill missing values
    if 'abstract' in df_cleaned.columns:
        df_cleaned['abstract'].fillna('', inplace=True)
    if 'authors' in df_cleaned.columns:
        df_cleaned['authors'].fillna('Unknown', inplace=True)
    if 'journal' in df_cleaned.columns:
        df_cleaned['journal'].fillna('Unknown', inplace=True)
    
    # Remove rows with missing title
    if 'title' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['title'].notna()]
    
    # Convert date and extract year
    if 'publish_time' in df_cleaned.columns:
        df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
        df_cleaned['publication_year'] = df_cleaned['publish_time'].dt.year
    
    # Create new columns
    if 'abstract' in df_cleaned.columns:
        df_cleaned['abstract_word_count'] = df_cleaned['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
    if 'title' in df_cleaned.columns:
        df_cleaned['title_word_count'] = df_cleaned['title'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
    
    return df_cleaned

# Load and prepare data
df = load_data()

if df is not None:
    df_cleaned = prepare_data(df)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Get year range
    if 'publication_year' in df_cleaned.columns:
        min_year = int(df_cleaned['publication_year'].min())
        max_year = int(df_cleaned['publication_year'].max())
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_year, max_year, (min_year, max_year)
        )
        df_filtered = df_cleaned[
            (df_cleaned['publication_year'] >= year_range[0]) & 
            (df_cleaned['publication_year'] <= year_range[1])
        ]
    else:
        df_filtered = df_cleaned
    
    # Journal filter
    st.sidebar.subheader("Filter by Journal")
    if 'journal' in df_filtered.columns:
        journals = df_filtered['journal'].unique()
        selected_journals = st.sidebar.multiselect(
            "Select Journals (leave empty for all)",
            journals,
            default=[]
        )
        if selected_journals:
            df_filtered = df_filtered[df_filtered['journal'].isin(selected_journals)]
    
    # Display statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Statistics")
    st.sidebar.metric("Total Papers", len(df_filtered))
    st.sidebar.metric("Original Dataset", len(df))
    if 'abstract_word_count' in df_filtered.columns:
        avg_words = df_filtered['abstract_word_count'].mean()
        st.sidebar.metric("Avg Abstract Words", f"{avg_words:.0f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "📚 Top Journals", "🔤 Word Analysis", "📊 Visualizations", "📋 Data"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", len(df_filtered))
        with col2:
            if 'publication_year' in df_filtered.columns:
                st.metric("Year Range", f"{int(df_filtered['publication_year'].min())}-{int(df_filtered['publication_year'].max())}")
        with col3:
            if 'journal' in df_filtered.columns:
                st.metric("Unique Journals", df_filtered['journal'].nunique())
        with col4:
            if 'authors' in df_filtered.columns:
                st.metric("Unique Authors", df_filtered['authors'].nunique())
        
        st.markdown("---")
        
        # Publications by year chart
        if 'publication_year' in df_filtered.columns:
            st.subheader("Publications Over Time")
            papers_by_year = df_filtered['publication_year'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(papers_by_year.index, papers_by_year.values, marker='o', linewidth=2.5, color='#2E86AB', markersize=8)
            ax.fill_between(papers_by_year.index, papers_by_year.values, alpha=0.3, color='#2E86AB')
            ax.set_title('Number of COVID-19 Publications Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Publication Year', fontsize=12)
            ax.set_ylabel('Number of Papers', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Tab 2: Top Journals
    with tab2:
        st.subheader("Top Publishing Journals")
        
        num_journals = st.slider("Number of top journals to display", 5, 30, 15)
        
        if 'journal' in df_filtered.columns:
            top_journals = df_filtered['journal'].value_counts().head(num_journals)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(top_journals)), top_journals.values, color='#A23B72')
            ax.set_yticks(range(len(top_journals)))
            ax.set_yticklabels(top_journals.index, fontsize=10)
            ax.set_title(f'Top {num_journals} Journals Publishing COVID-19 Research', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Papers', fontsize=12)
            ax.invert_yaxis()
            st.pyplot(fig)
            
            # Display as table
            st.subheader("Journal Statistics")
            journal_stats = pd.DataFrame({
                'Journal': top_journals.index,
                'Paper Count': top_journals.values,
                'Percentage': (top_journals.values / top_journals.sum() * 100).round(2)
            })
            st.dataframe(journal_stats, use_container_width=True)
    
    # Tab 3: Word Analysis
    with tab3:
        st.subheader("Most Frequent Words in Paper Titles")
        
        num_words = st.slider("Number of top words to display", 10, 50, 20)
        
        if 'title' in df_filtered.columns:
            all_titles = ' '.join(df_filtered['title'].astype(str)).lower()
            
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                         'covid', '19', 'coronavirus', 'sars', 'cov', 'covid-19', 'novel',
                         'new', 'study', 'research', 'analysis', 'case', 'effect', 'related'}
            
            words = [word for word in all_titles.split() if word.isalpha() and len(word) > 3 and word not in stop_words]
            word_freq = Counter(words)
            top_words = word_freq.most_common(num_words)
            
            words_list, counts_list = zip(*top_words)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(top_words)), counts_list, color='#F18F01')
            ax.set_xticks(range(len(top_words)))
            ax.set_xticklabels(words_list, rotation=45, ha='right', fontsize=10)
            ax.set_title(f'Top {num_words} Most Frequent Words in Paper Titles', fontsize=14, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12)
            st.pyplot(fig)
    
    # Tab 4: Visualizations
    with tab4:
        st.subheader("Advanced Visualizations")
        
        col1, col2 = st.columns(2)
        
        # Abstract word count distribution
        with col1:
            if 'abstract_word_count' in df_filtered.columns:
                st.subheader("Abstract Word Count Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df_filtered['abstract_word_count'], bins=50, color='#06A77D', edgecolor='black')
                ax.set_title('Distribution of Abstract Word Counts', fontsize=12, fontweight='bold')
                ax.set_xlabel('Word Count', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                st.pyplot(fig)
        
        # Title word count distribution
        with col2:
            if 'title_word_count' in df_filtered.columns:
                st.subheader("Title Word Count Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df_filtered['title_word_count'], bins=30, color='#FF006E', edgecolor='black')
                ax.set_title('Distribution of Title Word Counts', fontsize=12, fontweight='bold')
                ax.set_xlabel('Word Count', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                st.pyplot(fig)
        
        # Word Cloud
        st.subheader("Word Cloud of Paper Titles")
        if 'title' in df_filtered.columns:
            all_titles_wordcloud = ' '.join(df_filtered['title'].astype(str))
            wordcloud = WordCloud(width=1200, height=600, background_color='white', 
                                 colormap='viridis', max_words=100).generate(all_titles_wordcloud)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    # Tab 5: Data Table
    with tab5:
        st.subheader("Filtered Dataset Preview")
        
        # Select columns to display
        available_cols = ['title', 'abstract', 'authors', 'journal', 'publication_year', 'abstract_word_count']
        display_cols = [col for col in available_cols if col in df_filtered.columns]
        
        st.dataframe(df_filtered[display_cols].head(100), use_container_width=True)
        
        # Download button
        csv = df_filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="covid19_filtered_data.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.info("🔍 **Tips:** Use the sidebar controls to filter data by year and journal. Click tabs to explore different analyses.")

else:
    st.error("Failed to load data. Please ensure metadata.csv is in the correct location.")