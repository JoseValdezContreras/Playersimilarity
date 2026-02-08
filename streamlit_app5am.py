"""
Soccer Player Style Clustering Analysis
Machine Learning Portfolio Project

Author: Jose V
Dataset: 400k+ shots from European leagues (2014-2022)
Source: Understat
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import VarianceThreshold
from mplsoccer import PyPizza
from matplotlib import patheffects

# Page configuration
st.set_page_config(
    page_title="Soccer Player Style Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .cluster-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #d62728;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and processing
@st.cache_data
def load_and_process_data():
    """Load and process the soccer shot data"""
    df = pd.read_parquet('datacompleta.parquet')
    
    # Keep date info for later
    df_date = df[['player', 'year']].copy()
    
    # Select relevant columns
    df_main = df[['X', 'Y', 'xG', 'player', 'situation', 'shotType', 'match_id', 'GOAL']].copy()
    
    return df_main, df_date

@st.cache_data
def aggregate_player_stats(df):
    """Aggregate statistics by player"""
    df_player = df.groupby('player').agg(
        X_avg=('X', 'mean'),
        Y_std=('Y', 'std'),
        xG_avg=('xG', 'mean'),
        xG_sum=('xG', 'sum'),
        Goal_sum=('GOAL', 'sum'),
        total_shots=('situation', 'count'),
        Head_percent=('shotType', lambda x: (x == 'Head').mean()),
        Openplay_percent=('situation', lambda x: (x == 'OpenPlay').mean()),
        DirectFreekick_percent=('situation', lambda x: (x == 'DirectFreekick').mean()),
        DirectFreekick_goal_percent=('GOAL', lambda x: x[(df.loc[x.index, 'situation'] == 'DirectFreekick')].sum() / x.sum() if x.sum() > 0 else 0),
        Penalty_goal_percent=('GOAL', lambda x: x[(df.loc[x.index, 'situation'] == 'Penalty')].sum() / x.sum() if x.sum() > 0 else 0),
        Openplay_goal_percent=('GOAL', lambda x: x[(df.loc[x.index, 'situation'] == 'OpenPlay')].sum() / x.sum() if x.sum() > 0 else 0),
        Head_goal_conversion=('GOAL', lambda x: x[(df.loc[x.index, 'shotType'] == 'Head')].sum() / x.sum() if x.sum() > 0 else 0)
    ).reset_index()
    
    # Filter for minimum shots threshold
    MIN_SHOTS = 40
    df_player = df_player[df_player['total_shots'] >= MIN_SHOTS].copy()
    
    # Calculate xG overperformance
    df_player['avgxGoverperformance'] = (df_player['Goal_sum'] - df_player['xG_sum']) / df_player['total_shots']
    
    return df_player

@st.cache_data
def perform_clustering(df_player, n_clusters=6):
    """Perform cosine K-means clustering - EXACTLY as in temp.py"""
    # Prepare features (drop volume-based metrics) - EXACTLY as in temp.py
    cluster_features = df_player.drop(columns=['total_shots', 'xG_sum', 'Goal_sum'])
    
    # Remove low variance features - EXACTLY as in temp.py
   # p = 0.99
    #sel = VarianceThreshold(threshold=(p * (1 - p)))
    #sel.fit(cluster_features.drop('player', axis=1))
    #features_to_keep = cluster_features.drop('player', axis=1).columns[sel.get_support()]
    #features_to_drop = [column for column in cluster_features.columns if column not in features_to_keep and column != 'player']
    
    # Keep only selected features but preserve player column
    #cluster_features_for_clustering = cluster_features[['player'] + list(features_to_keep)]
    
    # CRITICAL: Use StandardScaler THEN Normalizer - EXACTLY as in temp.py
    scaler = StandardScaler()
    normalizer = Normalizer(norm='l2')
    
    cluster_features_indexed = cluster_features.set_index('player')
    X_scaled = scaler.fit_transform(cluster_features_indexed)
    X_cosine = normalizer.fit_transform(X_scaled)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cosine)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cosine)
    
    # t-SNE for better separation
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_cosine)
    
    return clusters, X_pca, X_tsne, cluster_features, pca, X_cosine

# Main app
def main():
    # Title
    st.markdown('<p class="main-header">⚽ Soccer Player Style Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Using Machine Learning to Identify and Classify Playing Styles</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df, df_date = load_and_process_data()
        df_player = aggregate_player_stats(df)
        clusters, X_pca, X_tsne, cluster_features, pca, X_cosine = perform_clustering(df_player)
        df_player['cluster'] = clusters
        
        # Add year data
        player_years = df_date.groupby('player').agg(
            first_seen=('year', 'min'),
            years_active=('year', 'nunique')
        ).reset_index()
        df_player = df_player.merge(player_years, on='player', how='left')
    
    # Check if user wants to see full story
    if 'show_full_story' not in st.session_state:
        st.session_state.show_full_story = False
    
    if not st.session_state.show_full_story:
        # Landing page: Interactive Scouting Tool
        show_landing_page_scouting(df_player, X_tsne, X_pca, X_cosine, pca, cluster_features)
    else:
        # Full story navigation
        show_full_story(df, df_player, X_pca, X_tsne, cluster_features, pca, X_cosine)

def show_full_story(df, df_player, X_pca, X_tsne, cluster_features, pca, X_cosine):
    """Show full story with navigation"""
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Add button to return to scouting tool
    if st.sidebar.button("← Back to Scouting Tool", type="primary"):
        st.session_state.show_full_story = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to:",
        ["🎯 Project Overview",
         "📊 Dataset Exploration", 
         "🤖 Understanding Cosine K-Means",
         "📏 Methodology & Features",
         "🔬 PCA & t-SNE Analysis",
         "👥 Cluster Profiles",
         "💡 Applications"]
    )
    
    # Page routing
    if page == "🎯 Project Overview":
        show_overview(df, df_player)
    elif page == "📊 Dataset Exploration":
        show_dataset_exploration(df, df_player)
    elif page == "🤖 Understanding Cosine K-Means":
        show_cosine_kmeans_explanation()
    elif page == "📏 Methodology & Features":
        show_methodology(df_player, cluster_features)
    elif page == "🔬 PCA & t-SNE Analysis":
        show_combined_pca_tsne(df_player, X_pca, X_tsne, pca, cluster_features)
    elif page == "👥 Cluster Profiles":
        show_cluster_profiles(df_player)
    elif page == "💡 Applications":
        show_applications()

def show_landing_page_scouting(df_player, X_tsne, X_pca, X_cosine, pca, cluster_features):
    """Landing page: Interactive scouting tool with t-SNE and pizza plots"""
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 Player Style Similarity Finder
    
    This tool uses machine learning to find players with similar attacking styles. Select any player to see:
    - Their position in style space (t-SNE visualization)
    - Top 5 most similar players based on attacking characteristics
    - Detailed style comparison using pizza charts
    """)
    
    st.markdown("---")
    
    # Player selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get some famous players for defaults
        famous_defaults = ['Lionel Messi', 'Cristiano Ronaldo', 'Harry Kane', 'Erling Haaland', 'Mohamed Salah', 'Kylian Mbappé']
        available_famous = [p for p in famous_defaults if p in df_player['player'].values]
        default_player = available_famous[0] if available_famous else df_player.nlargest(1, 'total_shots')['player'].iloc[0]
        
        selected_player = st.selectbox(
            "🎯 Select a player to analyze:",
            options=sorted(df_player['player'].unique()),
            index=sorted(df_player['player'].unique()).index(default_player)
        )
    
    with col2:
        min_year = st.number_input(
            "Min. first seen year:",
            min_value=int(df_player['first_seen'].min()),
            max_value=int(df_player['first_seen'].max()),
            value=2021,
            help="Filter for recent/emerging players"
        )
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    
    player_idx = df_player[df_player['player'] == selected_player].index[0]
    player_vector = X_cosine[df_player.index.get_loc(player_idx)]
    
    similarities = cosine_similarity([player_vector], X_cosine)[0]
    
    results = df_player.copy()
    results['similarity'] = similarities
    results = results[results['player'] != selected_player]
    results = results[results['first_seen'] >= min_year]
    
    top_similar = results.nlargest(5, 'similarity')
    
    # Create t-SNE visualization with selected player highlighted
    st.markdown('<p class="subsection-header">📊 Player Style Space (t-SNE Projection)</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # Plot all clusters
    for cluster in sorted(df_player['cluster'].unique()):
        mask = df_player['cluster'] == cluster
        cluster_data = df_player[mask]
        fig.add_trace(go.Scatter(
            x=X_tsne[mask, 0],
            y=X_tsne[mask, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=6, color=colors_map.get(cluster, '#000000'), opacity=0.4),
            text=cluster_data['player'],
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=True
        ))
    
    # Highlight selected player with star
    selected_tsne_pos = X_tsne[df_player.index.get_loc(player_idx)]
    fig.add_trace(go.Scatter(
        x=[selected_tsne_pos[0]],
        y=[selected_tsne_pos[1]],
        mode='markers+text',
        name=selected_player,
        marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='black')),
        text=[selected_player],
        textposition='top center',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate=f'<b>{selected_player}</b> (Selected)<extra></extra>',
        showlegend=False
    ))
    
    # Highlight top 5 similar players
    for idx, row in top_similar.iterrows():
        similar_pos = X_tsne[df_player.index.get_loc(idx)]
        fig.add_trace(go.Scatter(
            x=[similar_pos[0]],
            y=[similar_pos[1]],
            mode='markers',
            name=row['player'],
            marker=dict(size=12, color='red', symbol='circle', line=dict(width=2, color='darkred')),
            text=[row['player']],
            hovertemplate=f'<b>{row["player"]}</b><br>Similarity: {row["similarity"]:.3f}<extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        title='Player Style Clustering (t-SNE)',
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"⭐ **{selected_player}** is shown as a gold star. 🔴 Red circles are the 5 most similar players.")
    
    st.markdown("---")
    
    # Show top 5 similar players
    st.markdown('<p class="subsection-header">🎯 Top 5 Most Similar Players</p>', unsafe_allow_html=True)
    
    display_cols = ['player', 'similarity', 'cluster', 'total_shots', 'xG_avg', 'Goal_sum', 'first_seen', 'years_active']
    
    st.dataframe(
        top_similar[display_cols].style.format({
            'similarity': '{:.3f}',
            'total_shots': '{:.0f}',
            'xG_avg': '{:.3f}',
            'Goal_sum': '{:.0f}',
            'first_seen': '{:.0f}',
            'years_active': '{:.0f}'
        }).background_gradient(subset=['similarity'], cmap='RdYlGn', vmin=0.8, vmax=1.0),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Pizza plot comparison
    st.markdown('<p class="subsection-header">🍕 Detailed Style Comparison (Pizza Charts)</p>', unsafe_allow_html=True)
    
    compare_player = st.selectbox(
        "Select a player to compare in detail:",
        options=top_similar['player'].tolist(),
        index=0
    )
    
    if compare_player:
        create_pizza_comparison(df_player, selected_player, compare_player, cluster_features)
    
    st.markdown("---")
    
    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📖 Learn About the Methodology Behind This Tool", type="primary", use_container_width=True):
            st.session_state.show_full_story = True
            st.rerun()
    
    st.markdown("---")
    
    # Show overview visualizations below
    st.markdown('<p class="section-header">📊 Understanding the Clusters</p>', unsafe_allow_html=True)
    
    st.markdown("""
    The machine learning model identified **6 distinct attacking styles** in European soccer.
    Below you can see how these styles separate in 2D space using two different methods:
    """)
    
    # Side by side PCA and t-SNE with famous players marked
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PCA Projection**")
        st.caption("Linear dimensionality reduction - preserves global structure")
        fig_pca = create_labeled_pca_plot(df_player, X_pca, colors_map)
        st.pyplot(fig_pca)
    
    with col2:
        st.markdown("**t-SNE Projection**")
        st.caption("Non-linear - emphasizes cluster separation")
        fig_tsne = create_labeled_tsne_plot(df_player, X_tsne, colors_map)
        st.pyplot(fig_tsne)
    
    st.markdown("""
    <div class="insight-box">
    <b>Key Observation:</b> Both methods reveal distinct player groups, but t-SNE shows clearer
    separation between clusters. This validates that our features successfully captured distinct
    attacking styles. Famous players are labeled to give you reference points for each cluster.
    </div>
    """, unsafe_allow_html=True)

def create_pizza_comparison(df_player, player1_name, player2_name, cluster_features):
    """Create pizza plot comparison between two players"""
    
    # Get player data
    player1_data = df_player[df_player['player'] == player1_name].iloc[0]
    player2_data = df_player[df_player['player'] == player2_name].iloc[0]
    
    # Features for pizza plot (normalized percentages work best)
    pizza_features = [
        'xG_avg', 'X_avg', 'Y_std', 'Head_percent', 
        'Openplay_percent', 'DirectFreekick_percent', 'avgxGoverperformance'
    ]
    
    feature_labels = [
        'Shot Quality\n(xG avg)', 'Position\n(X avg)', 'Movement\n(Y std)', 
        'Headers\n(%)', 'Open Play\n(%)', 'Free Kicks\n(%)', 'Finishing\n(xG+/-)'
    ]
    
    # Get values
    player1_values = []
    player2_values = []
    
    for feature in pizza_features:
        player1_values.append(player1_data[feature])
        player2_values.append(player2_data[feature])
    
    # Normalize to 0-100 scale for each feature
    normalized_p1 = []
    normalized_p2 = []
    
    for i, feature in enumerate(pizza_features):
        min_val = df_player[feature].min()
        max_val = df_player[feature].max()
        
        if max_val - min_val > 0:
            p1_norm = ((player1_values[i] - min_val) / (max_val - min_val)) * 100
            p2_norm = ((player2_values[i] - min_val) / (max_val - min_val)) * 100
        else:
            p1_norm = 50
            p2_norm = 50
        
        normalized_p1.append(p1_norm)
        normalized_p2.append(p2_norm)
    
    # Create pizza plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f0f0f0')
    
    # Define colors
    slice_colors = ['#1f77b4'] * len(pizza_features)
    text_colors = ['#000000'] * len(pizza_features)
    
    # Create PyPizza object
    baker = PyPizza(
        params=feature_labels,
        background_color='#f0f0f0',
        straight_line_color='#000000',
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20
    )
    
    # Plot
    baker.make_pizza(
        normalized_p1,
        figsize=(8, 8.5),
        color_blank_space=['white'] * len(pizza_features),
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(edgecolor='#000000', zorder=2, linewidth=1),
        kwargs_params=dict(color='#000000', fontsize=11, va='center'),
        kwargs_values=dict(color='#000000', fontsize=11, zorder=3,
                          bbox=dict(edgecolor='#000000', facecolor='cornflowerblue',
                                   boxstyle='round,pad=0.2', lw=1))
    )
    
    # Add player 2 as comparison layer
    baker.make_pizza(
        normalized_p2,
        figsize=(8, 8.5),
        color_blank_space=['white'] * len(pizza_features),
        slice_colors=['#ff7f0e'] * len(pizza_features),
        value_colors=['#000000'] * len(pizza_features),
        value_bck_colors=['#ff7f0e'] * len(pizza_features),
        blank_alpha=0.2,
        kwargs_slices=dict(edgecolor='#000000', zorder=1, linewidth=1, alpha=0.6),
        kwargs_params=dict(color='#000000', fontsize=0),  # Hide labels for second player
        kwargs_values=dict(color='#000000', fontsize=0)   # Hide values for second player
    )
    
    # Add title
    fig.text(
        0.515, 0.97, f'{player1_name} vs {player2_name}', size=16,
        ha='center', color='#000000', weight='bold'
    )
    
    # Add subtitle
    similarity = cosine_similarity(
        [X_cosine[df_player[df_player['player'] == player1_name].index[0]]],
        [X_cosine[df_player[df_player['player'] == player2_name].index[0]]]
    )[0][0]
    
    fig.text(
        0.515, 0.94, f'Style Similarity: {similarity:.3f}', size=12,
        ha='center', color='#000000'
    )
    
    # Add legend
    fig.text(0.34, 0.02, f'● {player1_name}', size=11, color='#1f77b4', weight='bold', ha='right')
    fig.text(0.66, 0.02, f'● {player2_name}', size=11, color='#ff7f0e', weight='bold', ha='left')
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    <div class="insight-box">
    <b>How to Read Pizza Charts:</b>
    <ul>
        <li>Each slice represents a different attacking characteristic</li>
        <li>Larger slices = higher values for that metric (relative to all players)</li>
        <li>Similar shapes = similar playing styles</li>
        <li>Blue overlay = selected player | Orange overlay = comparison player</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def create_labeled_pca_plot(df_player, X_pca, colors_map):
    """Create PCA plot with famous players labeled"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all players by cluster
    for cluster in sorted(df_player['cluster'].unique()):
        mask = df_player['cluster'] == cluster
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors_map.get(cluster, '#000000'),
            label=f'Cluster {cluster}',
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Label famous players (top player from each cluster)
    famous_players = []
    for cluster in sorted(df_player['cluster'].unique()):
        cluster_players = df_player[df_player['cluster'] == cluster]
        # Get most prominent player (highest total shots)
        top_player = cluster_players.nlargest(1, 'total_shots').iloc[0]
        famous_players.append(top_player)
    
    # Add labels
    for player_data in famous_players:
        player_name = player_data['player']
        idx = df_player[df_player['player'] == player_name].index[0]
        pos = X_pca[df_player.index.get_loc(idx)]
        
        # Add text with outline for visibility
        text = ax.text(pos[0], pos[1], player_name, 
                      fontsize=9, fontweight='bold',
                      ha='center', va='bottom')
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title('PCA: Player Style Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_labeled_tsne_plot(df_player, X_tsne, colors_map):
    """Create t-SNE plot with famous players labeled"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all players by cluster
    for cluster in sorted(df_player['cluster'].unique()):
        mask = df_player['cluster'] == cluster
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colors_map.get(cluster, '#000000'),
            label=f'Cluster {cluster}',
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Label famous players (top player from each cluster)
    famous_players = []
    for cluster in sorted(df_player['cluster'].unique()):
        cluster_players = df_player[df_player['cluster'] == cluster]
        top_player = cluster_players.nlargest(1, 'total_shots').iloc[0]
        famous_players.append(top_player)
    
    # Add labels
    for player_data in famous_players:
        player_name = player_data['player']
        idx = df_player[df_player['player'] == player_name].index[0]
        pos = X_tsne[df_player.index.get_loc(idx)]
        
        # Add text with outline
        text = ax.text(pos[0], pos[1], player_name,
                      fontsize=9, fontweight='bold',
                      ha='center', va='bottom')
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE: Clear Cluster Separation', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def show_overview(df, df_player):
    """Page 1: Project Overview"""
    st.markdown('<p class="section-header">🎯 Project Goal</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Visualizing Player Style in Soccer
    
    This project aims to **quantify and classify attacking styles** of soccer players using machine learning. 
    Rather than simply measuring who scores the most goals or takes the most shots, we want to understand 
    **how players attack** — their unique fingerprint on the field.
    
    #### Key Questions:
    - Can we identify distinct attacking styles using shot data?
    - How do we separate "style" from "volume" or career length?
    - Can we find emerging players with similar profiles to established stars?
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Shots Analyzed", f"{len(df):,}")
    with col2:
        st.metric("Players (40+ shots)", f"{len(df_player):,}")
    with col3:
        st.metric("Years Covered", "2014-2022")
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Why This Matters</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    Traditional soccer analytics focus on <b>outputs</b> (goals, assists, shots) which favor:
    <ul>
        <li>Players with longer careers (more data)</li>
        <li>Players in dominant teams (more opportunities)</li>
        <li>Established stars over emerging talent</li>
    </ul>
    
    By analyzing <b>style</b> instead of volume, we can:
    <ul>
        <li>Compare players across different career stages</li>
        <li>Identify similar players regardless of team quality</li>
        <li>Find undervalued talent with elite characteristics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_combined_pca_tsne(df_player, X_pca, X_tsne, pca, cluster_features):
    """Combined PCA and t-SNE analysis page"""
    st.markdown('<p class="section-header">🔬 PCA & t-SNE: Visualizing High-Dimensional Styles</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Dimensionality Reduction for Visualization
    
    Our player style data exists in high-dimensional space (one dimension per feature).
    To visualize it in 2D, we use two complementary techniques:
    
    - **PCA (Principal Component Analysis)**: Linear method that preserves global structure
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear method that emphasizes local structure
    """)
    
    st.markdown("---")
    
    # Side by side comparison
    st.markdown('<p class="subsection-header">📊 Side-by-Side Comparison</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    with col1:
        st.markdown("**PCA Projection**")
        fig_pca = create_labeled_pca_plot(df_player, X_pca, colors_map)
        st.pyplot(fig_pca)
        
        st.markdown("""
        **PCA Characteristics:**
        - Linear transformation
        - Preserves global variance
        - Circular structure from StandardScaler + L2 normalization
        - Good for understanding feature relationships
        """)
    
    with col2:
        st.markdown("**t-SNE Projection**")
        fig_tsne = create_labeled_tsne_plot(df_player, X_tsne, colors_map)
        st.pyplot(fig_tsne)
        
        st.markdown("""
        **t-SNE Characteristics:**
        - Non-linear transformation
        - Emphasizes local neighborhoods
        - Clear cluster separation
        - Better for visual cluster identification
        """)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">🔍 What Do These Components Mean?</p>', unsafe_allow_html=True)
    
    # PCA loadings
    feature_names = cluster_features.drop('player', axis=1).columns
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Contributors to PC1:**")
        pc1_contrib = pca_components['PC1'].abs().sort_values(ascending=False).head(5)
        for feature, value in pc1_contrib.items():
            direction = "→" if pca_components.loc[feature, 'PC1'] > 0 else "←"
            st.write(f"{direction} **{feature}**: {abs(value):.3f}")
    
    with col2:
        st.markdown("**Top Contributors to PC2:**")
        pc2_contrib = pca_components['PC2'].abs().sort_values(ascending=False).head(5)
        for feature, value in pc2_contrib.items():
            direction = "↑" if pca_components.loc[feature, 'PC2'] > 0 else "↓"
            st.write(f"{direction} **{feature}**: {abs(value):.3f}")
    
    st.markdown("---")
    
    # Variance explained
    st.markdown('<p class="subsection-header">📈 Variance Explained</p>', unsafe_allow_html=True)
    
    variance_explained = pca.explained_variance_ratio_
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PC1 Variance", f"{variance_explained[0]:.1%}")
    with col2:
        st.metric("PC2 Variance", f"{variance_explained[1]:.1%}")
    with col3:
        st.metric("Total (2D)", f"{sum(variance_explained):.1%}")
    
    st.info(f"These 2 components capture {sum(variance_explained):.1%} of the total variance in player styles.")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="insight-box">
    <b>Key Takeaway:</b> The clear separation in t-SNE validates that our features successfully
    captured distinct attacking styles. PCA shows the circular structure from our StandardScaler + L2
    normalization approach, while t-SNE reveals the true "distance" between different playing styles.
    </div>
    """, unsafe_allow_html=True)

def show_dataset_exploration(df, df_player):
    """Page 2: Dataset Exploration"""
    st.markdown('<p class="section-header">📊 Dataset Deep Dive</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Data Source: Understat
    The dataset contains **400,000+ shot events** from European leagues between 2014-2022.
    """)
    
    st.markdown('<p class="subsection-header">Shot-Level Features</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Positional Data:**
        - `X`: Horizontal position on pitch
        - `Y`: Vertical position on pitch
        
        **Shot Quality:**
        - `xG`: Expected Goals (probability of scoring)
        - `GOAL`: Whether shot resulted in goal (0/1)
        """)
    
    with col2:
        st.markdown("""
        **Shot Context:**
        - `situation`: Open play, Penalty, Free kick, etc.
        - `shotType`: Head, Right foot, Left foot
        
        **Identifiers:**
        - `player`: Player name
        - `match_id`: Unique match identifier
        - `team`: Team name
        """)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Sample Shot Data</p>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Data Distribution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Shots per player distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        df_player['total_shots'].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Total Shots')
        ax.set_ylabel('Number of Players')
        ax.set_title('Distribution of Shots per Player (40+ shots)')
        ax.axvline(df_player['total_shots'].median(), color='red', linestyle='--', label=f'Median: {df_player["total_shots"].median():.0f}')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        # xG distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        df_player['xG_avg'].hist(bins=50, ax=ax, edgecolor='black', color='green', alpha=0.7)
        ax.set_xlabel('Average xG per Shot')
        ax.set_ylabel('Number of Players')
        ax.set_title('Distribution of Shot Quality (xG)')
        ax.axvline(df_player['xG_avg'].median(), color='red', linestyle='--', label=f'Median: {df_player["xG_avg"].median():.3f}')
        ax.legend()
        st.pyplot(fig)

def show_cosine_kmeans_explanation():
    """Page 3: Cosine K-Means Explanation"""
    st.markdown('<p class="section-header">🤖 Why Cosine K-Means?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### The Challenge: Separating Style from Volume
    
    Traditional clustering (like standard K-means) measures **Euclidean distance**, which is sensitive to magnitude.
    This creates a problem for our analysis:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <b>❌ Problem with Standard K-Means:</b><br><br>
        
        <b>Player A (Veteran):</b><br>
        - 300 shots over 8 years<br>
        - 150 from open play (50%)<br>
        - 30 headers (10%)<br><br>
        
        <b>Player B (Young):</b><br>
        - 50 shots over 1 year<br>
        - 25 from open play (50%)<br>
        - 5 headers (10%)<br><br>
        
        <b>Result:</b> Standard K-means sees them as DIFFERENT because 300 ≠ 50,
        even though their style (percentages) is identical!
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box" style="border-left-color: #2ca02c;">
        <b>✅ Solution with Cosine K-Means:</b><br><br>
        
        <b>Player A (Veteran):</b><br>
        - Style vector: [0.50, 0.10, ...]<br><br>
        
        <b>Player B (Young):</b><br>
        - Style vector: [0.50, 0.10, ...]<br><br>
        
        <b>Result:</b> Cosine distance measures the ANGLE between vectors,
        not magnitude. Same proportions = same style = SIMILAR!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">How Cosine Distance Works</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### How Cosine Distance Works (with Standardization)
    
    Our approach uses two normalization steps:
    1. **StandardScaler**: Standardizes features to zero mean and unit variance
    2. **L2 Normalizer**: Projects onto unit hypersphere for cosine distance
    
    Cosine distance then measures the angle between vectors, ignoring their length:
    
    $$
    \text{Cosine Similarity} = \\frac{A \cdot B}{||A|| \cdot ||B||}
    $$
    
    - **Value of 1**: Identical direction (same style)
    - **Value of 0**: Perpendicular (completely different styles)
    - **Value of -1**: Opposite directions
    """)
    
    # Visual demonstration
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw vectors
    origin = [0, 0]
    veteran = [3, 1]  # 300 shots, heavy open play
    young = [1, 0.33]  # 100 shots, same proportion
    different = [0.5, 2]  # Different style (more headers)
    
    ax.quiver(*origin, *veteran, angles='xy', scale_units='xy', scale=1, color='blue', width=0.01, label='Veteran (300 shots)')
    ax.quiver(*origin, *young, angles='xy', scale_units='xy', scale=1, color='green', width=0.01, label='Young Player (100 shots)')
    ax.quiver(*origin, *different, angles='xy', scale_units='xy', scale=1, color='red', width=0.01, label='Different Style')
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 3)
    ax.set_xlabel('Feature 1 (e.g., Open Play %)')
    ax.set_ylabel('Feature 2 (e.g., Header %)')
    ax.set_title('Cosine Distance: Measuring Angle, Not Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    st.pyplot(fig)
    
    st.markdown("""
    <div class="insight-box">
    <b>Key Insight:</b> The blue and green vectors point in the SAME DIRECTION (same style),
    even though the blue is longer (more career data). Cosine distance captures this!
    </div>
    """, unsafe_allow_html=True)

def show_methodology(df_player, cluster_features):
    """Page 4: Methodology & Features"""
    st.markdown('<p class="section-header">📏 Methodology: Defining "Style"</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Feature Engineering for Style Analysis
    
    To capture playing style independent of volume, we carefully selected and engineered features
    that describe **how** a player shoots, not **how much** they shoot.
    """)
    
    st.markdown('<p class="subsection-header">Selected Features</p>', unsafe_allow_html=True)
    
    feature_descriptions = {
        'X_avg': ('Shot Position (Horizontal)', 'Average horizontal position of shots. Lower values = shots from wider positions, Higher = central'),
        'Y_std': ('Shot Position Variation', 'Standard deviation of vertical position. Higher = varied shot locations, Lower = consistent positioning'),
        'xG_avg': ('Shot Quality', 'Average expected goals per shot. Higher = better shooting positions/chances'),
        'Head_percent': ('Aerial Threat', 'Percentage of shots taken with head. Higher = more aerial ability'),
        'Openplay_percent': ('Open Play Shots', 'Percentage of shots from open play (vs set pieces). Measures dynamic attacking'),
        'DirectFreekick_percent': ('Free Kick Specialist', 'Percentage of shots from direct free kicks'),
        'avgxGoverperformance': ('Clinical Finishing', 'Goals minus xG per shot. Positive = outperforming expectations'),
        'DirectFreekick_goal_percent': ('Free Kick Goals', 'Percentage of goals from free kicks'),
        'Penalty_goal_percent': ('Penalty Goals', 'Percentage of goals from penalties'),
        'Openplay_goal_percent': ('Open Play Goals', 'Percentage of goals from open play'),
        'Head_goal_conversion': ('Header Efficiency', 'Percentage of goals scored via headers')
    }
    
    for feature in cluster_features.columns:
        if feature in feature_descriptions:
            title, description = feature_descriptions[feature]
            with st.expander(f"📊 **{title}** (`{feature}`)"):
                st.write(description)
                
                # Show distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                df_player[feature].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel(feature)
                ax.set_ylabel('Count')
                ax.set_title(f'Distribution of {title}')
                st.pyplot(fig)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Why Team Shot Normalization Matters</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>The Problem:</b> A player taking 5 shots per match could mean two very different things:
    <ul>
        <li>Player on a dominant team (where the team takes 25 shots per match) = 20% involvement</li>
        <li>Player on a defensive team (where the team takes 10 shots per match) = 50% involvement</li>
    </ul>
    
    <b>The Solution:</b> We calculate the total number of shots each player's team took across all
    their matches, counting each match only once (even if the player took multiple shots).
    
    This gives us a <b>team-normalized shot involvement</b> metric that captures how central
    a player is to their team's attack, independent of team quality or playing style.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Features Excluded (and Why)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **❌ Excluded Volume Metrics:**
        - `total_shots`: Raw shot count (biased toward long careers)
        - `xG_sum`: Total expected goals (same issue)
        - `Goal_sum`: Total goals scored (same issue)
        
        These capture **output**, not **style**.
        """)
    
    with col2:
        st.markdown("""
        **✅ Why Percentages & Averages Work:**
        - `Head_percent`: 10% headers is 10% regardless of career length
        - `xG_avg`: Quality per shot, not total quality
        - `avgxGoverperformance`: Efficiency per shot
        
        These capture **how**, not **how much**.
        """)

def show_pca_analysis(df_player, X_pca, pca, cluster_features):
    """Page 5: PCA Visualization"""
    st.markdown('<p class="section-header">🔬 PCA: Visualizing High-Dimensional Styles</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is PCA?
    
    **Principal Component Analysis (PCA)** reduces our multi-dimensional player feature space
    down to 2 dimensions for visualization, while preserving as much variance as possible.
    
    - **PC1 (Principal Component 1)**: The direction of maximum variance in the data
    - **PC2 (Principal Component 2)**: The second-most variance, perpendicular to PC1
    """)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Interactive PCA Plot</p>', unsafe_allow_html=True)
    
    # Create interactive plotly figure
    fig = go.Figure()
    
    colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    for cluster in sorted(df_player['cluster'].unique()):
        mask = df_player['cluster'] == cluster
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=8, color=colors_map.get(cluster, '#000000'), opacity=0.6),
            text=df_player[mask]['player'],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Player Clusters in 2D Space (PCA)',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        hovermode='closest',
        width=900,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">What Do These Components Mean?</p>', unsafe_allow_html=True)
    
    # PCA loadings
    feature_names = cluster_features.columns
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Contributors to PC1:**")
        pc1_contrib = pca_components['PC1'].abs().sort_values(ascending=False).head(5)
        for feature, value in pc1_contrib.items():
            direction = "→" if pca_components.loc[feature, 'PC1'] > 0 else "←"
            st.write(f"{direction} **{feature}**: {abs(value):.3f}")
    
    with col2:
        st.markdown("**Top Contributors to PC2:**")
        pc2_contrib = pca_components['PC2'].abs().sort_values(ascending=False).head(5)
        for feature, value in pc2_contrib.items():
            direction = "↑" if pca_components.loc[feature, 'PC2'] > 0 else "↓"
            st.write(f"{direction} **{feature}**: {abs(value):.3f}")
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Interpreting the U-Shape</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    You may notice a U-shape or horseshoe pattern in the PCA plot. This is a common phenomenon called
    the <b>"horseshoe effect"</b> and it tells us something important:
    
    <ul>
        <li><b>There's a dominant gradient in the data</b>: Players likely vary along one main dimension
        (e.g., shot quality, attacking involvement, or positional style)</li>
        <li><b>PC2 captures non-linear variations</b>: The second component picks up quadratic relationships
        along this same gradient</li>
        <li><b>This is normal for compositional data</b>: Since we're using percentages (which sum to 100%),
        these constraints naturally create horseshoe patterns in PCA</li>
    </ul>
    
    <b>Important:</b> This doesn't affect our clustering! K-means operates on the full feature space,
    not just these 2 PCA dimensions. The PCA plot is only for visualization.
    </div>
    """, unsafe_allow_html=True)
    
    # Variance explained
    st.markdown('<p class="subsection-header">Variance Explained</p>', unsafe_allow_html=True)
    
    variance_explained = pca.explained_variance_ratio_
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PC1 Variance", f"{variance_explained[0]:.1%}")
    with col2:
        st.metric("PC2 Variance", f"{variance_explained[1]:.1%}")
    with col3:
        st.metric("Total (2D)", f"{sum(variance_explained):.1%}")
    
    st.info(f"These 2 components capture {sum(variance_explained):.1%} of the total variance in player styles.")

def show_tsne_analysis(df_player, X_tsne):
    """Page 6: t-SNE Analysis"""
    st.markdown('<p class="section-header">🎨 t-SNE: Revealing True Cluster Separation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Why t-SNE After PCA?
    
    While PCA is great for understanding which features drive variance, **t-SNE** (t-Distributed Stochastic
    Neighbor Embedding) is specifically designed to reveal cluster structure by:
    
    - Preserving **local neighborhood relationships**
    - Amplifying separation between distinct groups
    - Revealing the true "distance" between different playing styles
    """)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Interactive t-SNE Plot</p>', unsafe_allow_html=True)
    
    # Create interactive plotly figure
    fig = go.Figure()
    
    colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    cluster_names = {
        0: 'Cluster 0',
        1: 'Cluster 1', 
        2: 'Cluster 2',
        3: 'Cluster 3',
        4: 'Cluster 4',
        5: 'Cluster 5'
    }
    
    for cluster in sorted(df_player['cluster'].unique()):
        mask = df_player['cluster'] == cluster
        fig.add_trace(go.Scatter(
            x=X_tsne[mask, 0],
            y=X_tsne[mask, 1],
            mode='markers',
            name=cluster_names[cluster],
            marker=dict(size=8, color=colors_map.get(cluster, '#000000'), opacity=0.7),
            text=df_player[mask]['player'],
            hovertemplate='<b>%{text}</b><br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Player Style Clusters (t-SNE Projection)',
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        hovermode='closest',
        width=900,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Key Observations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>What t-SNE Reveals:</b>
    
    <ul>
        <li><b>Clear Cluster Separation</b>: Unlike PCA's horseshoe, t-SNE shows distinct "islands"
        of players with similar styles</li>
        <li><b>Style Diversity</b>: The distance between clusters indicates how different
        the playing styles are from each other</li>
        <li><b>Cluster Density</b>: Tight clusters suggest well-defined styles, while spread-out
        clusters indicate more variation within that style category</li>
        <li><b>Validation</b>: The fact that clusters separate cleanly in t-SNE space confirms
        that our features successfully captured distinct playing styles</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **PCA vs t-SNE:**
        - **PCA**: Linear, preserves global structure
        - **t-SNE**: Non-linear, emphasizes local structure
        - **Best practice**: Use both for complete picture
        """)
    
    with col2:
        st.markdown("""
        **What This Means:**
        - Clusters that overlap in PCA may separate in t-SNE
        - This confirms our styles are genuinely distinct
        - t-SNE better reflects "true" similarity between players
        """)

def show_cluster_profiles(df_player):
    """Page 7: Cluster Profiles"""
    st.markdown('<p class="section-header">👥 Cluster Profiles: The 6 Attacking Styles</p>', unsafe_allow_html=True)
    
    st.markdown("""
    After analyzing the characteristics of each cluster with K=6, we can examine
    the attacking style each group represents. You can customize the names and descriptions
    based on the characteristics you observe in your data.
    """)
    
    # Define cluster characteristics based on your analysis (K=6)
    cluster_info = {
        0: {
            'name': 'Cluster 0',
            'description': 'Attacking style identified by clustering analysis',
            'characteristics': ['Distinct attacking patterns', 'Specific shot selection tendencies', 'Unique positional characteristics']
        },
        1: {
            'name': 'Cluster 1',
            'description': 'Attacking style identified by clustering analysis',
            'characteristics': ['Distinct attacking patterns', 'Specific shot selection tendencies', 'Unique positional characteristics']
        },
        2: {
            'name': 'Cluster 2',
            'description': 'Attacking style identified by clustering analysis',
            'characteristics': ['Distinct attacking patterns', 'Specific shot selection tendencies', 'Unique positional characteristics']
        },
        3: {
            'name': 'Cluster 3',
            'description': 'Attacking style identified by clustering analysis',
            'characteristics': ['Distinct attacking patterns', 'Specific shot selection tendencies', 'Unique positional characteristics']
        },
        4: {
            'name': 'Cluster 4',
            'description': 'Attacking style identified by clustering analysis',
            'characteristics': ['Distinct attacking patterns', 'Specific shot selection tendencies', 'Unique positional characteristics']
        },
        5: {
            'name': 'Cluster 5',
            'description': 'Attacking style identified by clustering analysis',
            'characteristics': ['Distinct attacking patterns', 'Specific shot selection tendencies', 'Unique positional characteristics']
        }
    }
    
    # Cluster selection
    selected_cluster = st.selectbox(
        "Select a cluster to explore:",
        options=sorted(df_player['cluster'].unique()),
        format_func=lambda x: f"Cluster {x}: {cluster_info[x]['name']}"
    )
    
    st.markdown(f'<p class="cluster-name">{cluster_info[selected_cluster]["name"]}</p>', unsafe_allow_html=True)
    st.markdown(f"*{cluster_info[selected_cluster]['description']}*")
    
    st.markdown("**Key Characteristics:**")
    for char in cluster_info[selected_cluster]['characteristics']:
        st.markdown(f"- {char}")
    
    st.markdown("---")
    
    # Cluster statistics
    cluster_players = df_player[df_player['cluster'] == selected_cluster]
    
    st.markdown('<p class="subsection-header">Cluster Statistics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Players in Cluster", len(cluster_players))
    with col2:
        st.metric("Avg xG per Shot", f"{cluster_players['xG_avg'].mean():.3f}")
    with col3:
        st.metric("Avg Shot Position", f"{cluster_players['X_avg'].mean():.1f}")
    with col4:
        st.metric("Avg Headers %", f"{cluster_players['Head_percent'].mean():.1%}")
    
    st.markdown("---")
    
    # Top players in cluster
    st.markdown('<p class="subsection-header">Notable Players in This Cluster</p>', unsafe_allow_html=True)
    
    top_players = cluster_players.nlargest(15, 'total_shots')[
        ['player', 'total_shots', 'xG_avg', 'Goal_sum', 'avgxGoverperformance', 'first_seen', 'years_active']
    ]
    
    st.dataframe(
        top_players.style.format({
            'total_shots': '{:.0f}',
            'xG_avg': '{:.3f}',
            'Goal_sum': '{:.0f}',
            'avgxGoverperformance': '{:.3f}',
            'first_seen': '{:.0f}',
            'years_active': '{:.0f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Cluster comparison
    st.markdown('<p class="subsection-header">How This Cluster Compares</p>', unsafe_allow_html=True)
    
    # Calculate cluster averages
    cluster_stats = df_player.groupby('cluster')[['xG_avg', 'X_avg', 'Head_percent', 'Openplay_percent', 'avgxGoverperformance']].mean()
    
    comparison_metrics = {
        'Shot Quality (xG avg)': 'xG_avg',
        'Shot Position (X avg)': 'X_avg',
        'Header %': 'Head_percent',
        'Open Play %': 'Openplay_percent',
        'xG Overperformance': 'avgxGoverperformance'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, metric_col) in enumerate(comparison_metrics.items()):
        if idx < len(axes):
            ax = axes[idx]
            values = cluster_stats[metric_col]
            colors = ['#d62728' if i == selected_cluster else '#1f77b4' for i in values.index]
            ax.bar(values.index, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.set_xticks(values.index)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_applications():
    """Page 8: Applications"""
    st.markdown('<p class="section-header">💡 Real-World Applications</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This style-based clustering analysis has numerous practical applications in modern soccer:
    """)
    
    # Application 1
    st.markdown('<p class="subsection-header">1. Talent Identification & Scouting</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <b>The Use Case:</b><br>
        A club wants to replace their aging star striker (e.g., Harry Kane) but has a limited budget.
        Instead of competing for established names, they can:
        
        <ul>
            <li>Identify Kane's cluster (Elite Strikers)</li>
            <li>Find younger players in the same cluster</li>
            <li>Filter by recent seasons (emerging talent)</li>
            <li>Compare market value vs. style similarity</li>
        </ul>
        
        <b>Result:</b> Discover undervalued players with similar attacking profiles to elite strikers,
        but at a fraction of the cost.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.info("**💰 Value**: Scouts can identify 'stylistic replacements' rather than just 'statistical replacements', leading to better team fit.")
    
    # Application 2
    st.markdown('<p class="subsection-header">2. Squad Planning & Tactical Balance</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Understanding Your Squad:</b><br>
    
    Coaches can analyze their team's attacking options by cluster:
    <ul>
        <li><b>Cluster Distribution</b>: Do we have a balanced attack or are we too reliant on one style?</li>
        <li><b>Tactical Flexibility</b>: Can we adapt our style by substituting different cluster players?</li>
        <li><b>Injury Vulnerability</b>: If our elite striker gets injured, do we have a similar backup?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Application 3
    st.markdown('<p class="subsection-header">3. Player Development Pathways</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Young Players:**
        - Identify which elite cluster they most resemble
        - Study the characteristics of successful players in that cluster
        - Create targeted training programs to develop those skills
        """)
    
    with col2:
        st.markdown("""
        **For Transitioning Players:**
        - Track cluster movement as players age or change roles
        - Anticipate natural style evolution
        - Adapt positioning and responsibilities accordingly
        """)
    
    # Application 4
    st.markdown('<p class="subsection-header">4. Transfer Market Efficiency</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box" style="border-left-color: #2ca02c;">
    <b>Market Inefficiencies:</b><br>
    
    Players from lesser-known leagues or teams may be:
    <ul>
        <li>Stylistically similar to elite players in top leagues</li>
        <li>Significantly cheaper due to league/team reputation</li>
        <li>Overlooked by traditional scouting approaches</li>
    </ul>
    
    <b>Example:</b> A player in the Portuguese league with an "Elite Striker" profile might be
    available for €20M, while a player with similar style from the Premier League costs €80M.
    </div>
    """, unsafe_allow_html=True)
    
    # Application 5
    st.markdown('<p class="subsection-header">5. Opposition Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Understanding opponents' attacking styles helps defensive preparation:
    - **Poachers** → Tight marking in the box, cut passing lanes
    - **Wide Forwards** → Compact defensive shape, limit space in wide areas
    - **Aerial Threats** → Height at set pieces, prevent crosses
    - **Specialists** → Dedicated free-kick/penalty defense strategies
    """)
    
    st.markdown("---")
    
    st.markdown('<p class="subsection-header">Why This Approach Works</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⚖️ Fair Comparison**
        
        Normalizing by style (not volume) allows fair comparison across:
        - Different career stages
        - Different team qualities
        - Different leagues
        """)
    
    with col2:
        st.markdown("""
        **🎯 Predictive Power**
        
        Style is more stable than output:
        - Less affected by teammates
        - Less affected by tactics
        - More transferable across contexts
        """)
    
    with col3:
        st.markdown("""
        **💡 Actionable Insights**
        
        Provides concrete guidance:
        - Who to scout
        - How to train
        - When to sell/buy
        """)

def show_scouting_tool(df_player, X_cosine):
    """Page 9: Scouting Tool"""
    st.markdown('<p class="section-header">🔍 Scouting Tool: Find Similar Players</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Use this tool to find players with similar attacking profiles. This is useful for:
    - **Recruitment**: Finding replacements or alternatives to target players
    - **Development**: Identifying role models for young players
    - **Analysis**: Understanding which elite players a prospect most resembles
    """)
    
    st.markdown("---")
    
    # Player selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_player = st.selectbox(
            "Select a player to analyze:",
            options=sorted(df_player['player'].unique()),
            index=sorted(df_player['player'].unique()).index('Lionel Messi') if 'Lionel Messi' in df_player['player'].values else 0
        )
    
    with col2:
        n_similar = st.slider("Number of similar players to show:", 5, 20, 10)
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_year = st.number_input(
                "Minimum first seen year:",
                min_value=int(df_player['first_seen'].min()),
                max_value=int(df_player['first_seen'].max()),
                value=int(df_player['first_seen'].min())
            )
        
        with col2:
            max_year = st.number_input(
                "Maximum first seen year:",
                min_value=int(df_player['first_seen'].min()),
                max_value=int(df_player['first_seen'].max()),
                value=int(df_player['first_seen'].max())
            )
        
        with col3:
            same_cluster_only = st.checkbox("Same cluster only", value=False)
    
    # Find similar players
    if st.button("🔍 Find Similar Players", type="primary"):
        
        # Get player data
        player_idx = df_player[df_player['player'] == selected_player].index[0]
        player_vector = X_cosine[df_player.index.get_loc(player_idx)]
        player_cluster = df_player.loc[player_idx, 'cluster']
        
        # Calculate cosine similarity to all other players
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity([player_vector], X_cosine)[0]
        
        # Create results dataframe
        results = df_player.copy()
        results['similarity'] = similarities
        
        # Apply filters
        results = results[results['player'] != selected_player]  # Exclude the selected player
        results = results[results['first_seen'] >= min_year]
        results = results[results['first_seen'] <= max_year]
        
        if same_cluster_only:
            results = results[results['cluster'] == player_cluster]
        
        # Sort by similarity
        results = results.nlargest(n_similar, 'similarity')
        
        # Display results
        st.markdown(f'<p class="subsection-header">Players Most Similar to {selected_player}</p>', unsafe_allow_html=True)
        
        # Player stats card
        player_stats = df_player[df_player['player'] == selected_player].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cluster", f"{player_stats['cluster']}")
        with col2:
            st.metric("Total Shots", f"{player_stats['total_shots']:.0f}")
        with col3:
            st.metric("xG avg", f"{player_stats['xG_avg']:.3f}")
        with col4:
            st.metric("Goals", f"{player_stats['Goal_sum']:.0f}")
        
        st.markdown("---")
        
        # Similar players table
        display_cols = ['player', 'similarity', 'cluster', 'total_shots', 'xG_avg', 'Goal_sum', 
                       'Head_percent', 'Openplay_percent', 'avgxGoverperformance', 'first_seen', 'years_active']
        
        st.dataframe(
            results[display_cols].style.format({
                'similarity': '{:.3f}',
                'total_shots': '{:.0f}',
                'xG_avg': '{:.3f}',
                'Goal_sum': '{:.0f}',
                'Head_percent': '{:.1%}',
                'Openplay_percent': '{:.1%}',
                'avgxGoverperformance': '{:.3f}',
                'first_seen': '{:.0f}',
                'years_active': '{:.0f}'
            }).background_gradient(subset=['similarity'], cmap='RdYlGn', vmin=0.8, vmax=1.0),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Similarity interpretation
        st.markdown('<p class="subsection-header">Interpreting Similarity Scores</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Cosine Similarity Ranges:</b>
        <ul>
            <li><b>0.95 - 1.00</b>: Almost identical playing style</li>
            <li><b>0.90 - 0.95</b>: Very similar, excellent alternative</li>
            <li><b>0.85 - 0.90</b>: Similar style, good match</li>
            <li><b>0.80 - 0.85</b>: Some similarities, worth investigating</li>
            <li><b>< 0.80</b>: Different styles</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature comparison
        if len(results) > 0:
            st.markdown('<p class="subsection-header">Feature Comparison</p>', unsafe_allow_html=True)
            
            # Select a player to compare
            compare_player = st.selectbox(
                "Select a similar player to compare in detail:",
                options=results['player'].tolist()
            )
            
            if compare_player:
                # Get comparison data
                player_data = df_player[df_player['player'] == selected_player].iloc[0]
                compare_data = df_player[df_player['player'] == compare_player].iloc[0]
                
                # Create comparison chart
                comparison_features = ['xG_avg', 'X_avg', 'Head_percent', 'Openplay_percent', 
                                      'DirectFreekick_percent', 'avgxGoverperformance']
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for idx, feature in enumerate(comparison_features):
                    ax = axes[idx]
                    
                    values = [player_data[feature], compare_data[feature]]
                    colors = ['#1f77b4', '#ff7f0e']
                    bars = ax.bar([selected_player, compare_player], values, color=colors, alpha=0.7, edgecolor='black')
                    
                    ax.set_ylabel(feature)
                    ax.set_title(feature)
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
