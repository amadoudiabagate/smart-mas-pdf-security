# upma.py

import pandas as pd
from collections import defaultdict

class UserProfileManagementAgent:
    def __init__(self):
        # Dictionary to store each user's prediction history
        self.user_history = defaultdict(list)

    def update_history(self, df_results):
        """
        Update the prediction history of users from a DataFrame.
        
        Parameters:
        - df_results (pd.DataFrame): must contain columns ['ID_Utilisateur', 'Prediction']
        """
        for _, row in df_results.iterrows():
            user = row["ID_Utilisateur"]
            prediction = row["Prediction"]  # 1 = malicious, 0 = clean
            self.user_history[user].append(prediction)

    def update_profile(self, user_id, prediction):
        """
        Update the profile of a specific user based on the latest prediction.

        Parameters:
        - user_id (str/int): The ID of the user
        - prediction (int): 0 for clean, 1 for malicious

        Returns:
        - str: "suspicious" if more than 50% of files are malicious (and at least 5 files),
               "clean" otherwise,
               "insufficient_data" if not enough data
        """
        # Create a one-row DataFrame as expected by update_history
        df = pd.DataFrame([{
            "ID_Utilisateur": user_id,
            "Prediction": prediction
        }])
        self.update_history(df)

        # Compute the user's malicious rate
        predictions = self.user_history[user_id]
        if len(predictions) >= 5:
            malicious_rate = sum(predictions) / len(predictions)
            return "suspicious" if malicious_rate > 0.5 else "clean"
        else:
            return "insufficient_data"

    def analyze_user_profiles(self):
        """
        Analyze user profiles to detect those with suspicious behavior.

        Returns:
        - list: User IDs flagged as suspicious
        """
        suspicious_users = []
        for user, predictions in self.user_history.items():
            if len(predictions) >= 5:
                malicious_rate = sum(predictions) / len(predictions)
                if malicious_rate > 0.5:
                    suspicious_users.append(user)
        return suspicious_users

    def generate_user_statistics(self):
        """
        Generate a summary table of user behavior statistics.

        Returns:
        - pd.DataFrame: with columns ['ID_Utilisateur', 'Total_Files', 'Malicious_Files', 'Malicious_Rate']
        """
        data = []
        for user, predictions in self.user_history.items():
            total = len(predictions)
            malicious_count = sum(predictions)
            rate = malicious_count / total if total > 0 else 0
            data.append({
                "ID_Utilisateur": user,
                "Total_Files": total,
                "Malicious_Files": malicious_count,
                "Malicious_Rate": round(rate, 2)
            })
        return pd.DataFrame(data)
