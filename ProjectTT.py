import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_exam_data(file_path):
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return str(e), None, None, None, None

    # Ensure columns exist
    if 'Attempt_Time_Minutes' not in df.columns or 'Score' not in df.columns:
        return "Error: CSV must have 'Attempt_Time_Minutes' and 'Score' columns.", None, None, None, None

    # --- PART 1: STATISTICAL ANALYSIS ---
    # Basic Descriptive Statistics
    stats = df[['Attempt_Time_Minutes', 'Score']].describe().reset_index()

    # Correlation
    correlation = df['Attempt_Time_Minutes'].corr(df['Score'])
    corr_msg = f"Pearson Correlation: {correlation:.4f}"
    if abs(correlation) < 0.2: corr_msg += " (No significant relationship)"
    elif correlation > 0: corr_msg += " (Positive trend: More time ≈ Higher score)"
    else: corr_msg += " (Negative trend: More time ≈ Lower score)"

    # Efficiency (Score per Minute)
    df['Efficiency'] = df['Score'] / df['Attempt_Time_Minutes']
    
    # --- PART 2: SEGMENTATION (QUADRANT ANALYSIS) ---
    # We use median as the cutoff to categorize students
    med_time = df['Attempt_Time_Minutes'].median()
    med_score = df['Score'].median()

    def categorize(row):
        if row['Score'] >= med_score and row['Attempt_Time_Minutes'] < med_time:
            return 'Fast High-Achiever'
        elif row['Score'] >= med_score and row['Attempt_Time_Minutes'] >= med_time:
            return 'Diligent High-Achiever'
        elif row['Score'] < med_score and row['Attempt_Time_Minutes'] < med_time:
            return 'Rusher (Low Score)'
        else:
            return 'Struggler (Low Score/Slow)'

    df['Category'] = df.apply(categorize, axis=1)
    category_counts = df['Category'].value_counts().reset_index()
    category_counts.columns = ['Student Type', 'Count']

    # --- PART 3: VISUALIZATIONS ---
    
    # FIG 1: Distribution & Outliers (Boxplots & Histograms)
    fig1 = plt.figure(figsize=(10, 8))
    
    # Score Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['Score'], kde=True, color='skyblue')
    plt.title("Score Distribution")
    
    # Time Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['Attempt_Time_Minutes'], kde=True, color='orange')
    plt.title("Time Distribution")
    
    # Score Outliers
    plt.subplot(2, 2, 3)
    sns.boxplot(x=df['Score'], color='skyblue')
    plt.title("Score Outliers (Boxplot)")
    
    # Time Outliers
    plt.subplot(2, 2, 4)
    sns.boxplot(x=df['Attempt_Time_Minutes'], color='orange')
    plt.title("Time Outliers (Boxplot)")
    
    plt.tight_layout()

    # FIG 2: Segmentation Scatter Plot (Quadrant Analysis)
    fig2 = plt.figure(figsize=(10, 6))
    
    # Scatter plot colored by Category
    sns.scatterplot(
        data=df, 
        x='Attempt_Time_Minutes', 
        y='Score', 
        hue='Category', 
        style='Category', 
        s=100, 
        palette='viridis'
    )
    
    # Draw Median Lines to show the quadrants
    plt.axvline(med_time, color='red', linestyle='--', label=f'Median Time ({med_time:.1f})')
    plt.axhline(med_score, color='blue', linestyle='--', label=f'Median Score ({med_score:.1f})')
    
    # Add a trend line (Regression)
    sns.regplot(data=df, x='Attempt_Time_Minutes', y='Score', scatter=False, color='gray', line_kws={'alpha':0.5})

    plt.title("Student Segmentation (Quadrant Analysis)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return stats, category_counts, corr_msg, fig1, fig2

def calc_efficiency(score, time):
    if time <= 0: return "Time must be > 0"
    eff = score / time
    return f"Efficiency: {eff:.2f} points/min"

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("# 📊 Complete Exam Analysis (Standard Libraries)")
    gr.Markdown("Upload your CSV file to perform distribution, outlier, and segmentation analysis.")
    
    with gr.Tab("Analysis Dashboard"):
        with gr.Row():
            file_input = gr.File(label="Upload CSV", type="filepath")
            btn_analyze = gr.Button("Analyze Data", variant="primary")
        
        # Text Summary
        out_corr = gr.Textbox(label="Correlation Insight")
        
        # Data Tables
        with gr.Row():
            out_stats = gr.Dataframe(label="General Statistics")
            out_cat = gr.Dataframe(label="Student Segments (Counts)")
        
        # Plots
        with gr.Row():
            out_plot_dist = gr.Plot(label="Distributions & Outliers")
        with gr.Row():
            out_plot_seg = gr.Plot(label="Quadrant Analysis (Segmentation)")

    with gr.Tab("Efficiency Calculator"):
        with gr.Row():
            in_score = gr.Number(label="Score")
            in_time = gr.Number(label="Time (Minutes)")
        btn_calc = gr.Button("Calculate")
        out_res = gr.Textbox(label="Result")
        btn_calc.click(calc_efficiency, [in_score, in_time], out_res)

    btn_analyze.click(
        analyze_exam_data, 
        inputs=file_input, 
        outputs=[out_stats, out_cat, out_corr, out_plot_dist, out_plot_seg]
    )

if __name__ == "__main__":
    demo.launch()
