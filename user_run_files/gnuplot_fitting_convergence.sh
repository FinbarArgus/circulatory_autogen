#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./generate_plot.sh <param_id_ouput_dir>"
    exit 1
fi

# Assign the input CSV file to a variable
csv_file="$1/best_cost_history.csv"

# Create a GNU Plot script file
gnuplot_script="generate_plot.gp"

num_columns=$(head -n 1 "$csv_file" | tr ',' '\n' | wc -l)

# Create or overwrite the GNU Plot script
# set terminal pngcairo enhanced font 'Arial,12' size 800,600
cat > "$gnuplot_script" <<EOL
set terminal wxt enhanced
set datafile separator ","
set xlabel "Iteration"
set ylabel "cost"
set title "Your Plot Title"
plot for [i=1:$num_columns] "$csv_file" using 0:i with linespoints title "cost"
EOL

# Generate the plot using GNU Plot
gnuplot -persist "$gnuplot_script"

# Clean up the temporary script
rm "$gnuplot_script"

echo "cost history plot generated"

# Now plot the parameters
# Assign the input CSV file to a variable
csv_file_2="$1/best_param_vals_history.csv"

# Create a GNU Plot script file
gnuplot_script_2="generate_plot_2.gp"

num_cols_2=$(head -n 1 "$csv_file_2" | tr ',' '\n' | wc -l)

column_names_2=$(head -n 1 "$csv_file_2" | tr ',' '\n' | sed 's/"//g')

# Create or overwrite the GNU Plot script
# set terminal pngcairo enhanced font 'Arial,12' size 800,600
cat > "$gnuplot_script_2" <<EOL
set terminal wxt enhanced
set datafile separator ","
set xlabel "Iteration"
set ylabel "normed param vals"
set title "Your Plot Title"
plot for [i=1:$num_cols_2] "$csv_file_2" using 0:i with linespoints title columnheader(i)
EOL

# Generate the plot using GNU Plot
gnuplot -persist "$gnuplot_script_2"

# Clean up the temporary script
rm "$gnuplot_script_2"

echo "Param Vals plot generated"
