from train import train_model
from test import test_model
import torch
from model import LSTMWithMLP
import optuna

def main():

    study = optuna.create_study(direction='minimize')
    study.optimize(train_model, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")




    # Train the model
    # model = train_model()

    # model = LSTMWithMLP(lstm_input_size=1, output_size=1, num_gaussians=5, mlp_input_dim=11)
    # state_path = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\处理数据\final_model_state.pth"
    # model.load_state_dict(torch.load(state_path))

    # Test the model
    # average_nll_loss, average_mae_loss, average_mape_loss, average_r2_score = test_model(model)

    # # # Output the average losses and R^2 score
    # print(f'Average NLL Loss on Validation Set: {average_nll_loss:.4f}')
    # # print(f'Average RMSE on Test Set: {average_rmse_loss:.4f}')
    # print(f'Average MAE on Validation Set: {average_mae_loss:.4f}')
    # print(f'Average MAPE on Validation Set: {average_mape_loss:.4f}')
    # print(f'R^2 Score on Validation Set: {average_r2_score:.4f}')

    # # Save the model state
    # state_path = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\处理数据\final_model_state.pth"
    # torch.save(model.state_dict(), state_path)
    # print("Model state saved successfully.")

if __name__ == "__main__":
    main()
