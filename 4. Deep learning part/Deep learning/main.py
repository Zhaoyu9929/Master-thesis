from data_preprocessing import prepare_lstm_data, prepare_mlp_data
from train import train_model
from visualization import plot_training_loss

def main():
    # Load the lstm_data and mlp_data
    lstm_data = prepare_lstm_data()
    mlp_data = prepare_mlp_data()
    
    # Train the model
    model = train_model(lstm_data, mlp_data)

    # Visualize the training loss
    plot_training_loss(train_model.train_losses)


if __name__ == "__main__":
    main()
