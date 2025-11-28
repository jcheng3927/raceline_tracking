"""
Script to transform x/y coordinates in racetrack CSV files.
Creates half y, double x, and double y versions of tracks.
"""

import csv
import os

def transform_coordinates(input_file, output_file, x_factor=1.0, y_factor=1.0):
    """
    Read a racetrack CSV file, transform coordinates, and save to a new file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        x_factor: Multiplier for x coordinates
        y_factor: Multiplier for y coordinates
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
                if len(parts) >= 2:
                    try:
                        # Transform the x coordinate (first column)
                        x = float(parts[0])
                        parts[0] = str(x * x_factor)
                        # Transform the y coordinate (second column)
                        y = float(parts[1])
                        parts[1] = str(y * y_factor)
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
    
    # Define transformations: (suffix, x_factor, y_factor)
    transformations = [
        ('halfy', 1.0, 0.5),
        ('doublex', 2.0, 1.0),
        ('doubley', 1.0, 2.0),
    ]
    
    for track in tracks:
        for suffix, x_factor, y_factor in transformations:
            # Process main track file
            input_file = os.path.join(base_dir, f'{track}.csv')
            output_file = os.path.join(base_dir, f'{track}_{suffix}.csv')
            
            # Check if input file exists
            if os.path.exists(input_file):
                print(f'Processing {track} ({suffix})...')
                transform_coordinates(input_file, output_file, x_factor, y_factor)
                print(f'  Created: {output_file}')
            else:
                print(f'Warning: {input_file} not found, skipping.')
            
            # Process raceline file
            raceline_input = os.path.join(base_dir, f'{track}_raceline.csv')
            raceline_output = os.path.join(base_dir, f'{track}_raceline_{suffix}.csv')
            
            if os.path.exists(raceline_input):
                print(f'Processing {track} raceline ({suffix})...')
                transform_coordinates(raceline_input, raceline_output, x_factor, y_factor)
                print(f'  Created: {raceline_output}')
            else:
                print(f'Warning: {raceline_input} not found, skipping.')
    
    print('\nDone! All files processed.')

if __name__ == '__main__':
    main()
