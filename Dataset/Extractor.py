import cv2
import numpy as np
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_file_info(path):
    file_info_list = []
    for foldername , _ , filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(foldername , filename)
            
            file_info_list.append(
                {
                    'name':filename,
                    'path':filepath
                }
            )
    return file_info_list

def extract_from_16bit(path):
    file_info_list = get_file_info(path=path)
    
    def process_and_save(fileinfo):
        filename = fileinfo['name']
        filepath = fileinfo['path']
        
        img = cv2.imread(filepath , cv2.IMREAD_UNCHANGED)

    