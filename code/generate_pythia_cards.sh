filename="a_b_samples_uniform_1.csv"

# Check if the file exists
if [ -f "$filename" ]; then
    # Set the IFS to handle commas as the field separator
    IFS=','

    # Read the CSV file line by line
    while read -r col1 col2 col3; do
        
        # Process each column as needed
        echo "row index: $col1, a: $col2, b: $col3"
        
    done < "$filename"

    # Reset IFS to its default value
    IFS=$' \t\n'
else
    echo "File not found: $filename"
fi