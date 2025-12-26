import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_exam_data(file_path):
    # Load the dataset directly from the file path
    df = pd.read_csv(file_path)
    
    # 1. Descriptive Statistics for Time and Score
    # reset_index() converts the index (mean, std, min...) into a column for display
    stats = df[['Attempt_Time_Minutes', 'Score']].describe().reset_index()
    
    # 2. Calculate Pearson Correlation
    correlation = df['Attempt_Time_Minutes'].corr(df['Score'])
    corr_result = f"Pearson Correlation Coefficient: {correlation:.4f}"
    
    # 3. Analyze Efficiency (Score per Minute)
    df['Efficiency'] = df['Score'] / df['Attempt_Time_Minutes']
    eff_stats = df['Efficiency'].describe().reset_index()
    
    # 4. Create Scatter Plot
    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Attempt_Time_Minutes', y='Score', data=df)
    plt.title(f"Attempt Time vs Score (Corr: {correlation:.2f})")
    plt.xlabel("Attempt Time (Minutes)")
    plt.ylabel("Score")
    plt.grid(True)
    
    return stats, fig, corr_result, eff_stats

def calc_efficiency(score, time):
    if time <= 0:
        return "Time must be greater than 0"
    return f"Efficiency Score: {score / time:.3f} points/min"

# Build the simplified User Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 Student Exam Analysis")
    
    with gr.Tab("Upload & Analyze"):
        gr.Markdown("Upload the student CSV file to see statistics and graphs.")
        
        # Input: File upload (returns the file path string)
        file_input = gr.File(label="Upload CSV File", type="filepath")
        btn_analyze = gr.Button("Analyze Data")
        
        # Outputs
        with gr.Row():
            out_stats = gr.Dataframe(label="Descriptive Statistics")
            out_plot = gr.Plot(label="Relationship Plot")
        
        with gr.Row():
            out_corr = gr.Textbox(label="Correlation Analysis")
            out_eff = gr.Dataframe(label="Efficiency Statistics")
            
        # Link button to function
        btn_analyze.click(
            analyze_exam_data, 
            inputs=file_input, 
            outputs=[out_stats, out_plot, out_corr, out_eff]
        )

    with gr.Tab("Simple Calculator"):
        gr.Markdown("Quickly check efficiency for a specific student.")
        with gr.Row():
            in_score = gr.Number(label="Score")
            in_time = gr.Number(label="Time (Minutes)")
        
        btn_calc = gr.Button("Calculate Efficiency")
        out_calc = gr.Textbox(label="Result")
        
        btn_calc.click(calc_efficiency, inputs=[in_score, in_time], outputs=out_calc)

# Launch the app
if __name__ == "__main__":
    demo.launch()