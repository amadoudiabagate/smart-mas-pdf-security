
from model import SimulationModel

def main():
    print("=== Multi-Agent Simulation for Malicious PDF Detection ===")
    print("Initializing model...")

    # Create the model with the desired parameters
    model = SimulationModel(num_files=400, num_users=10)

    print("Starting simulation...")
    model.run()

    print("Simulation finished.")
    print("Results have been saved to 'output/simulation_results_full.csv'.")

if __name__ == "__main__":
    main()
