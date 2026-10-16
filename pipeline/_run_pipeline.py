from src import data_prep, eda

def main():
    raw_data_path = 'data/train.csv'
    processed_data_path = 'data/processed/cleaned_data.csv'
    
    print("1. Starting Data Preparation...")
    data_prep.process_and_save(raw_data_path, processed_data_path)
    
    print("2. Generating EDA Graphs...")
    eda.generate_all_eda(processed_data_path, output_dir='output/eda_plots/')
    
    print("Pipeline Execution Complete. Check the processed data and EDA outputs.")

if __name__ == "__main__":
    main()