import pandas as pd
import os
import datetime

class DataManagementAgent:
    def __init__(self, dataset_path='data/initial_dataset.csv', archive_dir='data/archives'):
        self.dataset_path = dataset_path
        self.archive_dir = archive_dir
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Temp storage for results
        self.simulation_records = []

    def get_dataset(self):
        """Load the main dataset."""
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            self._verify_dataset_integrity(df)
            return df
        else:
            raise FileNotFoundError(f"The file {self.dataset_path} is not found.")

    def update_with_simulation(self, new_data: pd.DataFrame):
        """Append simulated data to the main dataset."""
        old_df = self.get_dataset()
        self._archive_existing_dataset()
        updated_df = pd.concat([old_df, new_data], ignore_index=True)
        updated_df.to_csv(self.dataset_path, index=False)
        print(f"{len(new_data)} lines added. New dataset saved.")
        self._log_update(len(new_data))

    def archive_simulation_results(self, simulation_data: pd.DataFrame):
        """Archives the simulation results in a separate file."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(self.archive_dir, f'simulation_{timestamp}.csv')
        simulation_data.to_csv(archive_path, index=False)
        print(f"Simulation results archived: {archive_path}")
        return archive_path

    def _verify_dataset_integrity(self, df: pd.DataFrame):
        """Checks that the dataset columns match expected format."""
        expected_columns = [
            'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
            'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
            'feature11', 'feature12', 'feature13', 'feature14', 'feature15',
            'feature16', 'feature17', 'feature18', 'feature19', 'feature20',
            'feature21', 'feature22', 'user_id', 'label'
        ]
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"The dataset is invalid. Missing columns: {missing}")

    def _archive_existing_dataset(self):
        """Backs up the current dataset before overwriting."""
        if os.path.exists(self.dataset_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.archive_dir, f'backup_{timestamp}.csv')
            os.rename(self.dataset_path, backup_path)
            print(f"Archived dataset: {backup_path}")

    def _log_update(self, rows_added):
        """Logs the dataset update."""
        log_path = os.path.join(self.archive_dir, "log.txt")
        with open(log_path, "a") as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {rows_added} lines added to the dataset.\n")

    def store(self, file: dict, prediction: int):
        """
        Store a single prediction result for potential later archival.

        Parameters:
        - file (dict): Original file metadata and features.
        - prediction (int): Model prediction (0 or 1).
        """
        record = file.copy()
        record["prediction"] = prediction
        self.simulation_records.append(record)
