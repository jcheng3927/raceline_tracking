"""
Script to halve x coordinates in racetrack CSV files.
Processes Monza, Montreal, and IMS track files, creating new versions
with x coordinates divided by 2.
"""

import csv
import os

def halve_x_coordinates(input_file, output_file):
    """
    Read a racetrack CSV file, halve the x coordinates, and save to a new file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    with open(output_file, 'w', newline='') as outfile:
        for line in lines:
            # Keep header/comment lines unchanged
            if line.startswith('#'):
                outfile.write(line)
            else:
                # Parse the CSV line
                parts = line.strip().split(',')
                if len(parts) >= 2:  # Changed from >= 4 to >= 2 to handle raceline files
                    try:
                        # Halve the x coordinate (first column)
                        x = float(parts[0])
                        parts[0] = str(x / 2.0)
                        # Write the modified line
                        outfile.write(','.join(parts) + '\n')
                    except ValueError:
                        # If conversion fails, keep the line unchanged
                        outfile.write(line)
                else:
                    # Keep malformed lines unchanged
                    outfile.write(line)

def main():
    # Define the tracks to process
    tracks = ['Monza', 'Montreal', 'IMS']
    
    # Base directory for racetrack files
    base_dir = './racetracks'
    
    for track in tracks:
        # Process main track file
        input_file = os.path.join(base_dir, f'{track}.csv')
        output_file = os.path.join(base_dir, f'{track}_halfx.csv')
        
        # Check if input file exists
        if os.path.exists(input_file):
            print(f'Processing {track}...')
            halve_x_coordinates(input_file, output_file)
            print(f'  Created: {output_file}')
        else:
            print(f'Warning: {input_file} not found, skipping.')
        
        # Process raceline file
        raceline_input = os.path.join(base_dir, f'{track}_raceline.csv')
        raceline_output = os.path.join(base_dir, f'{track}_raceline_halfx.csv')
        
        if os.path.exists(raceline_input):
            print(f'Processing {track} raceline...')
            halve_x_coordinates(raceline_input, raceline_output)
            print(f'  Created: {raceline_output}')
        else:
            print(f'Warning: {raceline_input} not found, skipping.')
    
    print('\nDone! All files processed.')

if __name__ == '__main__':
    main()
