import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_one_params_data(path):
    print(path)
    with open(path) as f:
        data = np.loadtxt(path, delimiter='\t', dtype=np.double)
    name = path.split('/')[-1].split(".")[0]
    return name, data


def walkFile(file, mode): # 若内存不足，需修改成读部分数据
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        name_list, data_list = [], []
        # 遍历文件
        for f in tqdm(files):
            path = os.path.join(root, f)
            if mode == "noisy":
                name, data = read_one_nosiy_data(path)
            elif mode == "params":
                name, data = read_one_params_data(path)
            name_list.append(name)
            data_list.append(data)
    return name_list, np.array(data_list)


def read_one_nosiy_data(path):
    with open(path) as f:
        content = f.read()
        content = content.replace("# star_temp: ","").replace("# star_logg: ","").replace("# star_rad: ","").replace("# star_mass: ","").replace("# star_k_mag: ","").replace("# period: ","")
        content = content.replace("\n", "\t")
        data = np.fromstring(content, sep='\t', dtype=np.double)
    name = path.split('/')[-1].split(".")[0]
    return name, data


def read_and_save():
    name_list, y_data_list = walkFile("training_set/params_train","params")
    df_y = pd.DataFrame(y_data_list)
    df_y["name"] = name_list

    name_list, data_list = walkFile("training_set/noisy_train", "noisy")
    df_X = pd.DataFrame(data_list)
    df_X["name"] = name_list

    name_list, data_list = walkFile("test_set/noisy_test", "noisy")
    df_test = pd.DataFrame(data_list)
    df_test["name"] = name_list

    return df_X, df_y, df_test
    