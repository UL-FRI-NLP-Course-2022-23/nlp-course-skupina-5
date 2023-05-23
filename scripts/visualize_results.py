import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


def visualize(result_path):
    manual_label, automatic_label = get_annotation(result_path)
    manual_label, automatic_label = np.array(manual_label), np.array(automatic_label)

    if len(manual_label) != len(automatic_label):
        print("Error sizes of stories mismatch!")
        return

    path_parts = result_path.replace(".csv", "").split("\\")
    title = path_parts[-1].replace("_", " ")

    x_values = np.array(list(range(1, len(manual_label) + 1)))

    true_x = x_values[manual_label == 1]
    true_auto = automatic_label[manual_label == 1]
    false_x = x_values[manual_label == 0]
    false_auto = automatic_label[manual_label == 0]

    # moving avg lowpass
    w = 7
    avg_auto = np.convolve(automatic_label, np.ones(7)/7, "same")

    # fourier lowpass
    N = len(automatic_label)
    print(N)
    t = np.linspace(0, 1, N, endpoint=False)
    X = fft(automatic_label)
    f0 = int(0.12 * N)
    X[np.abs(np.fft.fftfreq(N, t[1]-t[0])) > f0] = 0
    fourier_auto = ifft(X).real

    plt.plot(x_values, automatic_label, label="Automatic label", color="black", linewidth=1)
    plt.plot(true_x, true_auto, marker="o", color="lime", linestyle="", label="Annotated salient")
    plt.plot(false_x, false_auto, marker="x", color="red", linestyle="", label="Annotated non-salient")
    
    # moving avg lowpass
    #plt.plot(x_values, avg_auto, label="Moving average", color="blue")

    # fourier lowpass
    plt.plot(x_values, fourier_auto, label="Moving average", color="blue")


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='Input')
    args = parser.parse_args()

    # Fill lists with stories you would like to visualize
    # results = [r"..\results\ANG\the_mouse_the_bird_and_the_sausage_window.csv",
    #            r"..\results\ANG\RUMPELSTILTSKIN_window.csv",
    #            r"..\results\ANG\THE_WOLF_AND_THE_SEVEN_LITTLE_KIDS_window.csv"]

    # results = [r"..\results\SLO\misek_pticka_in_klobasa_window.csv",
    #            r"..\results\SLO\spicparkeljc_window.csv",
    #            r"..\results\SLO\volk_in_sedem_kozlickov_window.csv"]

    visualize(args.input_file)

    # for result in results:
    #     visualize(result)