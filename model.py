# Agent acronyms (as per manuscript)
# UIA: User Interface Agent — entry point; user interaction, file intake, output delivery.
# PLA: Planning Agent — coordinates internal processes and execution order.
# GAA: Generative Augmentation Agent — detects imbalance, runs CTGAN.
# MLA: Machine Learning Agent — supervised classification, evaluation, retraining.
# UPMA: User Profile Management Agent — maintains risk profiles and user histories.
# DMA: Data Management Agent — archives results, manages datasets for retraining/audit.

import os
import csv
import random
from agents.uia import  UserInterfaceAgent
from agents.pla import PlanningAgent
from agents.gaa import GenerativeAugmentationAgent
from agents.mla import MachineLearningAgent
from agents.upma import UserProfileManagementAgent
from agents.dma import DataManagementAgent

class SimulationModel:
    def __init__(self, num_files=400, num_users=10):
        self.num_files = num_files
        self.num_users = num_users
        
        # Define data paths for PlanningAgent
        self.raw_data_path = "data/raw_data.csv"
        self.generated_data_path = "data/generated_data.csv"
        self.simulation_input_path = "data/simulation_input.csv"
        self.results_path = "data/results.csv"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        self.agents = {
            "UIA":  UserInterfaceAgent(),
            "PLA": PlanningAgent(
                raw_data_path=self.raw_data_path,
                generated_data_path=self.generated_data_path,
                simulation_input_path=self.simulation_input_path,
                results_path=self.results_path
            ),
            "GAA": GenerativeAugmentationAgent(),
            "MLA": MachineLearningAgent(),
            "UPMA": UserProfileManagementAgent(),
            "DMA": DataManagementAgent()
        }
        self.results = []

    def run(self):
        # Step 1: Initial generation of queues by UIA
        files = self.agents["UIA"].generate_pdf_files(self.num_files, self.num_users)

        # Step 2: Dataset verification and augmentation if necessary (GAA)
        balanced_files = self.agents["GAA"].augment_if_needed(files)

        # Step 3: Training the ML model (MLA)
        self.agents["MLA"].train_model(balanced_files)

        # Step 4: Simulation for each PDF file
        for file in files:
            prediction, score, model_used = self.agents["MLA"].predict(file)
            profile_status = self.agents["UPMA"].update_profile(file["user_id"], prediction)

            result = {
                "user_id": file["user_id"],
                "file_id": file["file_id"],
                "prediction": prediction,
                "score": round(score, 3),
                "model_used": model_used,
                "true_label": file["true_label"],
                "is_correct": prediction == file["true_label"],
                "file_type": file.get("file_type", "unknown"),
                "risk_level": file.get("risk_level", "medium"),
                "file_size_kb": file.get("file_size_kb", 500),
                "source": file.get("source", "unknown"),
                "nb_features": file.get("nb_features", 22),
                "has_js": file.get("has_js", False),
                "has_embedded_files": file.get("has_embedded_files", False),
                "user_profile_status": profile_status
            }

            self.results.append(result)
            self.agents["DMA"].store(file, prediction)

        self.save_results_to_csv()

    def save_results_to_csv(self, filename="output/simulation_results_full.csv"):
        if not self.results:
            print("No results to save.")
            return
        # Create output folder if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        keys = list(self.results[0].keys())
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    model = SimulationModel()
    model.run()