import os

def getFilesPath(folder_path, file_type):
    '''
    get files which contain in assigned folder and match with type
    return:
        files path: array of strings
    '''
    result = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.' + file_type):
            result.append(os.path.join(folder_path, file_name))
    return result