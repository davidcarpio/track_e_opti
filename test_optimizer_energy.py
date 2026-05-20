from src.trajectory_optimizer import optimize_trajectory

result = optimize_trajectory("data/tracks/sem_2025_eu.csv")
print(f"Energy Aero: {result.energy_aero_Wh} Wh")
print(f"Energy Rolling: {result.energy_rolling_Wh} Wh")
print(f"Energy Grade Up: {result.energy_grade_Wh} Wh")
print(f"Energy Drivetrain Loss: {result.energy_drivetrain_loss_Wh} Wh")
print(f"Energy Mechanical Braking: {result.energy_mechanical_braking_Wh} Wh")
print(f"Energy Potential Grade: {result.energy_potential_grade_Wh} Wh")
print(f"Energy Potential Kinetic: {result.energy_potential_kinetic_Wh} Wh")
print(f"Energy Recovered Regen: {result.energy_recovered_regen_Wh} Wh")
