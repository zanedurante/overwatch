import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_hero_counts(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Create a bar plot of the string counts
    plt.bar(df['Hero'], df['Count'])
    plt.xlabel('Hero')
    plt.ylabel('Count')
    plt.title('Frequency of Hero in Top 500 NA (in top 3 most-played)')

    # Add count values on top of each bar
    for i, count in enumerate(df['Count']):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # Rotate the x-axis labels if needed
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_string_counts(csv_files):
    # Create a list to store the DataFrames for each CSV file
    dfs = []

    # Read each CSV file into a pandas DataFrame and store it in the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    merged_df = pd.concat(dfs).groupby('Hero').sum().reset_index()

    # Get unique categories from the merged DataFrame
    categories = sorted(merged_df['Hero'].tolist())

    # Set the width of each bar
    bar_width = 0.2

    # Generate evenly spaced x-axis positions for each category
    x_positions = np.arange(len(categories))

    csv_names = ["Asia", "EU", "NA"]

    # Create a bar plot for each CSV file
    for (i, df), legend_name in zip(enumerate(dfs), csv_names):
        
        # Create a dictionary to store the counts for each category
        category_counts = {category: 0 for category in categories}

        # Update the dictionary with the counts from the current CSV file
        for index, row in df.iterrows():
            category_counts[row['Hero']] = row['Count']

        # Calculate the x-axis positions for the bars of the current CSV file
        x_pos = x_positions + (i * bar_width)

        # Create the bars for the current CSV file
        plt.bar(x_pos, [category_counts[category] for category in categories],
                width=bar_width, label=legend_name)

        # Add count values on top of each bar
        for j, category in enumerate(categories):
            plt.text(x_pos[j], category_counts[category],
                     str(category_counts[category]), ha='center', va='bottom', fontsize=8)

     # Set the x-axis tick positions and labels
    plt.xticks(x_positions + bar_width, categories, rotation=45)

    # Set the axis labels and title
    plt.xlabel('Hero')
    plt.ylabel('Count')
    plt.title('Hero Frequency in Top 500 (in top 3 most-played)')

    # Add a legend
    plt.legend()

    plt.tight_layout()

    # Display the plot
    plt.show()

csv_files = ['stats/overwatch-top-500-05-2023-asia.csv', 
             'stats/overwatch-top-500-05-2023-eu.csv',
             'stats/overwatch-top-500-05-2023-na.csv']
plot_string_counts(csv_files)

# Comment the following lines when not using visualizer.py directly
#output_file = 'stats/2023-05-top-500-NA.csv'
#plot_hero_counts(output_file)