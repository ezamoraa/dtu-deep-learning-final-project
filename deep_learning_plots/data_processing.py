from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from icecream import ic

def get_paths(path: str, get_dirs=False):
    if not get_dirs: 
        paths = sorted(glob(path))
    else:
        paths_unfiltered = sorted(glob(path))
        paths = [path for path in paths_unfiltered if '.' not in path]
    return paths

def create_data_frame(log: str):
    df = pd.read_csv(log)
    return df

def get_mean_time(log: str) -> float:
    data_frame = create_data_frame(log)
    if not 'image' in log:
        mean: float = data_frame.iloc[:, 0].mean()
    else:
        mean: float = data_frame.iloc[:, 1].mean()
    return mean

def get_total_time(logs_file_paths: list[str]):
    time = 0
    for log in logs_file_paths[:-1]:
        mean_time = get_mean_time(log)
        time += mean_time
    return time 

def get_mean_snnrme(log:str): 
    data_frame = create_data_frame(log)
    mean_intensity = data_frame.iloc[-1, -3]
    mean_geometry = data_frame.iloc[-1, -4]
    return mean_intensity, mean_geometry

def get_bpp_range_image(bottle_neck, iterations):
    original_img_size = (1812, 32)
    compressed_latent_space = (114,2)
    bpp = compressed_latent_space[0]*compressed_latent_space[1]*bottle_neck*iterations / (original_img_size[0]*original_img_size[1])
    return bpp
    
def get_test_specs(directory: str):
    name = directory.split('_')
    for word in name:
        if 'i' in word:
            iterations = word[1:]
        if 'b' in word:
            bottle_neck = word[1:]
    return int(iterations), int(bottle_neck)


def main():
    tests = []
    bpps = []
    mgs = []
    mis = []
    times = []
    directories = get_paths('*', get_dirs=True)
    for directory in directories:
        logs_file_paths = get_paths(directory + '/*.csv')
        total_time = get_total_time(logs_file_paths)
        mean_intensity_snnmrse, mean_geometry_snnmrse = get_mean_snnrme(logs_file_paths[-1])
        #print(total_time, mean_geometry_snnmrse, mean_intensity_snnmrse)
        iterations, bottle_neck = get_test_specs(directory)
        bpp = get_bpp_range_image(bottle_neck, iterations)
        tests.append({'total_time': total_time,
                      'mean_geometry_snnmrse': mean_geometry_snnmrse,
                      'mean_intensity_snnmrse': mean_intensity_snnmrse,
                      'bpp': bpp}
                        )
        bpps.append(bpp)
        mgs.append(mean_geometry_snnmrse)
        mis.append(mean_intensity_snnmrse)
        times.append(total_time)
    ic(tests)
    #plt.axis((0, 5, 0, 0.5))
    plt.plot(bpps,times, 'mo')
    plt.xlabel('BPP')
    #plt.ylabel('snnmrse')
    plt.grid(True)
    plt.show()

        
    




if __name__ == "__main__":
    main()