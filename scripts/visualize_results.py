import csv
import numpy as np
import matplotlib.pyplot as plt


def visualize(result_path):
    manual_label, automatic_label = get_annotation(result_path)

    if len(manual_label) != len(automatic_label):
        print("Error sizes of stories mismatch!")
        return

    path_parts = result_path.replace(".csv", "").split("\\")
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
    annotation_manual = []
    annotation_automatic = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # Skip first row
        next(csv_reader)
        for row in csv_reader:
            difference = row[4]
            labeled = row[5]
            try:
                to_float = float(difference)
                annotation_automatic.append(float(to_float))
            except:
                print("Error with cvs number conversion")
            try:
                to_float = float(labeled)
                annotation_manual.append(float(to_float))
            except:
                print("Error with cvs number conversion")

    return np.array(annotation_manual), np.array(annotation_automatic)


if __name__ == "__main__":

    # Fill lists with stories you would like to visualize
    results = [r"..\results\ANG\the_mouse_the_bird_and_the_sausage_window.csv",
               r"..\results\ANG\RUMPELSTILTSKIN_window.csv",
               r"..\results\ANG\THE_WOLF_AND_THE_SEVEN_LITTLE_KIDS_window.csv"]

    # results = [r"..\results\SLO\misek_pticka_in_klobasa_window.csv",
    #            r"..\results\SLO\spicparkeljc_window.csv",
    #            r"..\results\SLO\volk_in_sedem_kozlickov_window.csv"]

    for result in results:
        visualize(result)