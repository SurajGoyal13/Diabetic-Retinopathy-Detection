from retina_model import dataset_summary


def main() -> None:
    summary = dataset_summary()
    print("Dataset scan complete")
    print(f"Total labeled retina images: {summary['image_count']}")
    for label, count in summary["class_counts"].items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
