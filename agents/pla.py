import pandas as pd
from .gaa import GenerativeAugmentationAgent
from .mla import MachineLearningAgent
from .upma import UserProfileManagementAgent
from .dma import DataManagementAgent

class PlanningAgent:
    def __init__(self, raw_data_path, generated_data_path, simulation_input_path, results_path):
        self.raw_data_path = raw_data_path
        self.generated_data_path = generated_data_path
        self.simulation_input_path = simulation_input_path
        self.results_path = results_path

        # Initialize the sub-agents (you may need to pass required arguments)
        self.agent_aag = GenerativeAugmentationAgent()
        self.agent_aml = MachineLearningAgent()
        self.agent_agpu = UserProfileManagementAgent()
        self.agent_agd = DataManagementAgent()

    def preprocessing_phase(self):
        print("[PLA] Phase 1: Dataset verification and augmentation if necessary...")
        df = pd.read_csv(self.raw_data_path)
        balanced_df = self.agent_aag.generer_data_equilibrees(df)
        balanced_df.to_csv(self.generated_data_path, index=False)
        print("[PLA] Generated data saved.")

    def training_phase(self):
        print("[PLA] Phase 2: Model training with balanced data...")
        df_train = pd.read_csv(self.generated_data_path)
        self.agent_aml.entrainer_modeles(df_train)
        print("[PLA] Training completed.")

    def simulation_phase(self):
        print("[PLA] Phase 3: User queue simulation...")
        df_sim = pd.read_csv(self.simulation_input_path)

        # Prediction by ML Agent
        predictions = self.agent_aml.predire(df_sim)

        # User profile enrichment by User Profile Management Agent
        predictions_with_profiles = self.agent_agpu.ajouter_info_utilisateur(predictions)

        # Data update by Data Management Agent
        self.agent_agd.maj_jeu_de_data(predictions_with_profiles, self.generated_data_path)

        # Save results (model.py handles final complete .csv generation)
        predictions_with_profiles.to_csv(self.results_path, index=False)
        print("[PLA] Simulation results saved.")

    def execute_pipeline(self):
        self.preprocessing_phase()
        self.training_phase()
        self.simulation_phase()
        print("[PLA] Simulation pipeline completed successfully.")