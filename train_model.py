from retina_model import train_and_save_model


def main() -> None:
    result = train_and_save_model()
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.metrics_path}")
    print(f"Train size: {result.train_size}")
    print(f"Test size: {result.test_size}")
    print(f"Validation accuracy: {result.accuracy:.4f}")


if __name__ == "__main__":
    main()
