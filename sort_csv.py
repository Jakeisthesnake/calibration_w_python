import csv

def sort_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        rows = sorted(reader, key=lambda x: int(x[0]))  # Sort numerically by the first column
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

if __name__ == "__main__":
    input_csv = "/home/jake/calibration_euroc_data_copy/mav0/cam0/data.csv"  # Change this to your input file name
    output_csv = "/home/jake/calibration_euroc_data_copy/mav0/cam0/data_s.csv"  # Change this to your desired output file name
    sort_csv(input_csv, output_csv)
    print(f"Sorted CSV saved as {output_csv}")
