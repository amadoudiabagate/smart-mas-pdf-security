# uia.py
import pandas as pd
import numpy as np
import os
import random

class UserInterfaceAgent:
    def __init__(self, output_path='data/incoming_requests.csv', num_files=400, num_users=10):
        self.output_path = output_path
        self.num_files = num_files
        self.num_users = num_users
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _generate_single_observation(self, user_id, file_index):
        """Generates a simulated PDF record represented by 22 variables + user_id."""
        features = np.random.rand(22)  # Floating values between 0 and 1
        obs = {f'feature{i+1}': features[i] for i in range(22)}
        
        # Add required fields for the simulation
        obs.update({
            'user_id': user_id,
            'file_id': f"file_{file_index}",
            'true_label': random.choice([0, 1]),  # 0 = benign, 1 = malicious
            'file_type': random.choice(["pdf", "doc", "xls"]),
            'risk_level': random.choice(["low", "medium", "high"]),
            'file_size_kb': random.randint(100, 2000),
            'source': random.choice(["email", "web", "usb"]),
            'nb_features': 22,
            'has_js': random.choice([True, False]),
            'has_embedded_files': random.choice([True, False])
        })
        return obs

    def generate_requests(self):
        """Generates a set of requests distributed among users."""
        observations = []
        user_ids = [f"user_{i+1}" for i in range(self.num_users)]
        file_counter = 1

        # Balanced distribution: files distributed among users
        for user_id in user_ids:
            for _ in range(self.num_files // self.num_users):
                obs = self._generate_single_observation(user_id, file_counter)
                observations.append(obs)
                file_counter += 1

        df = pd.DataFrame(observations)
        df.to_csv(self.output_path, index=False)
        print(f"{self.num_files} files generated and saved in: {self.output_path}")
        return df
    
    def generate_pdf_files(self, num_files, num_users):
        """Alternative method name for compatibility - returns list of dictionaries."""
        self.num_files = num_files
        self.num_users = num_users
        df = self.generate_requests()
        return df.to_dict('records')