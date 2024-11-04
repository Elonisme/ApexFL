import copy
import csv

import numpy

from plot import get_filenames_in_directory, load_csv


def get_diff(origin, defense):
    diff = defense - origin
    return diff


def get_efficient(origin, defense):
    num_ba = len(origin)
    mean_diff = []
    for i in range(0, num_ba):
        mean_diff.append(get_diff(float(origin[i]), float(defense[i])))
    mean_diff = sum(mean_diff) / len(mean_diff)
    return mean_diff

def get_mean(data):
    mean_diff = []
    for i in data:
        mean_diff = numpy.append(mean_diff, float(i))
    mean_diff = numpy.mean(mean_diff)
    return mean_diff


if __name__ == '__main__':
    directory = "../save/csv/"
    keywords = ["mobilenet", "cifar10", 'trigger', '200', "0.15", "0.3"]
    origin_defense = "fedavg"
    filenames = get_filenames_in_directory(directory, keywords)

    result_ma = {}
    result_ba = {}
    result_time = {}

    save_paths = []
    for filename in filenames:
        split_filename = filename.split("-")
        model_name = split_filename[0]
        data_name = split_filename[1]
        attack_name = split_filename[2]
        split_filename = filename.split("-")
        defense_name = split_filename[3]
        save_paths.append(f"../save/csv/efficient/{model_name}-{data_name}-{attack_name}-{defense_name}")
        data = load_csv(directory, filename)
        num_data = len(data)
        ma, ba, _, time = data
        # if defense_name == 'small_fltrust':
        #     ma_40 = []
        #     ba_40 = []
        #     time_40 = []
        #     for index, _ in enumerate(ma):
        #         if index % 5 == 0:
        #             print(index)
        #             ma_40.append(ma[index])
        #             ba_40.append(ba[index])
        #             time_40.append(time[index])
        #     ma = tuple(ma_40)
        #     ba = tuple(ba_40)
        #     time = tuple(time_40)

        epoch = [i * 5 for i in range(0, len(ma))]
        result_ma[str(defense_name)] = ma
        result_ba[str(defense_name)] = ba
        result_time[str(defense_name)] = time

    orange_ma = result_ma[origin_defense]
    orange_ba = result_ba[origin_defense]
    orange_time = result_time[origin_defense]

    # headers = ['Ma Improvement (%)', 'Ma mean', 'Ba Improvement (%)', 'Ba mean', 'Time Improvement (s)']
    headers = ['Ma Improvement (%)', 'Ba Improvement (%)', 'Time Improvement (s)']
    for i, key in enumerate(result_ma.keys()):
        if key == origin_defense:
            print(get_mean(result_ma[key]))
            print(get_mean(result_ba[key]))
            continue
        ma_efficient_value = get_efficient(orange_ma, result_ma[key])
        ba_efficient_value = get_efficient(orange_ba, result_ba[key])
        time_efficient_value = get_efficient(orange_time, result_time[key])
        data = [[ma_efficient_value,
                 # get_mean(result_ma[key]),
                 ba_efficient_value,
                 # get_mean(result_ba[key]),
                 time_efficient_value]]
        print(f"{key} defense : {data}")
        csv_path = save_paths[i] + f"-{origin_defense}"
        with open(f'{csv_path}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)
        print("CSV文件保存成功！")
