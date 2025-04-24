#!/usr/bin/env python3
import pandas as pd
import random
import datetime
import os

def main():
    # Path to the sample submission file
    sample_submission_path = "../SampleSubmission.csv"
    
    # Check if file exists
    if not os.path.exists(sample_submission_path):
        sample_submission_path = "../SampleSubmission.csv"
        if not os.path.exists(sample_submission_path):
            # Try absolute path
            sample_submission_path = "/Users/adjei/Desktop/Amini Cocoa Contamination Challenge/SampleSubmission.csv"
            if not os.path.exists(sample_submission_path):
                print(f"Error: SampleSubmission.csv not found. Please check the path.")
                return
    
    # Read the sample submission file
    print(f"Reading sample submission from: {sample_submission_path}")
    df = pd.read_csv(sample_submission_path)
    
    # Classes to randomly assign
    classes = ['healthy', 'anthracnose', 'cssvd']
    
    # Randomly assign classes
    print("Varying class values between healthy, MOD, and FRT...")
    df['class'] = df['Image_ID'].apply(lambda x: random.choice(classes))
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create new filename with timestamp
    output_filename = f"../submission_varied_{timestamp}.csv"
    
    # Save the modified dataframe to a new CSV file
    print(f"Saving modified submission to: {output_filename}")
    df.to_csv(output_filename, index=False)
    
    print(f"Done! Modified submission saved as: {output_filename}")
    
    # Print class distribution statistics
    class_counts = df['class'].value_counts()
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} ({count/len(df)*100:.2f}%)")
    
if __name__ == "__main__":
    main()