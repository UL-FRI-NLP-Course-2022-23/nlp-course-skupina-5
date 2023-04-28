import csv
import numpy as np
import matplotlib.pyplot as plt


def visulaize(manual_path, automatic_path):
    manual_label = get_annotation(manual_path)
    automatic_label = get_annotation(automatic_path)

    manual_label = np.delete(manual_label, -1)

    print(manual_label)
    print(automatic_label)

    if len(manual_label) != len(automatic_label):
        print("Error sizes of stories mismatch!")
        return

    path_parts = manual_path.replace(".csv", "").split("\\")
    title = path_parts[-1].replace("_", " ")

    x_values = list(range(1, len(manual_label) + 1))

    plt.plot(x_values, manual_label, marker="s", label="Manual label", linestyle='')
    plt.plot(x_values, automatic_label, label="Automatic label")

    plt.xlabel("Sentence")
    plt.ylabel("Event Salience")
    plt.title("Event salience for story " + title)

    plt.legend()
    plt.show()


def get_annotation(file_path):
    annotation = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # Ship first row
        next(csv_reader)
        for row in csv_reader:
            if "results" in file_path:
                value = row[4]
            else:
                value = row[0]
            try:
                to_float = float(value)
                annotation.append(float(to_float))
            except:
                print("Error with cvs number conversion")
    return np.array(annotation)


if __name__ == "__main__":

    # Fill lists with stories you would like to compare
    manual_labeled = [r"..\data\labeled\misek_pticka_in_klobasa.csv"]
    automatic_labeled = [r"..\results\SLO\misek_pticka_in_klobasa.csv"]

    for i, j in zip(manual_labeled, automatic_labeled):
        visulaize(i, j)