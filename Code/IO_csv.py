import pandas as pd
import numpy as np
import os
import csv


def Input_train_data(shuffle_row=0):
    train_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dataset', 'train.csv') # ../Dataset/train.csv
    #print(test_file_path, train_file_path, submission_file_path)

    data=pd.read_csv(train_file_path)
    data_array=np.array(data)
    data_array[np.isnan(data_array)] = 0
    if shuffle_row==1:
        np.random.shuffle(data_array)
    # print(data_array, np.size(data_array, 0), "*",np.size(data_array, 1))
    x_data=data_array[:, :512]
    y_data=data_array[:, 512:]
    # print('x_data', x_data, "\n", 'y_data', y_data)
    return (x_data, y_data)

def Input_test_data():
    test_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dataset', 'test.csv') # ../Dataset/test.csv

    data=pd.read_csv(test_file_path)
    data_array=np.array(data)
    data_array[np.isnan(data_array)] = 0

    return data_array

def Output_ans(ans): # 要求ans 已经带有了编号
    ans_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dataset', 'ans.csv') # ../Dataset/submission.csv
    with open(ans_file_path, "w", encoding="utf-8") as f:
        csv_writer=csv.writer(f)
        name = ['Id', 'Label']
        csv_writer.writerow(name)
        csv_writer.writerows(ans)
        print('[Info]: Write to ans.csv finished.')
        f.close()