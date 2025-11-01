


from shutil import copyfile, copytree
import datetime, os
import pickle

class Dict2Class(object):
    '''
    Turns a dictionary into a class
    '''
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
    def update(self, newdict={}, **kwargs):
        self.__dict__.update(newdict, **kwargs)


def copy_file(filename, save_path):
    """
    copy file to desired location
    """
    if not os.path.isfile(save_path+filename):
	    copyfile(filename,save_path+os.path.basename(filename))

def copy_dir(filename, save_path):
    """
    copy dir to desired location
    """
    copytree(filename,save_path+os.path.basename(filename))

def create_save_dir(path, type="now"):
    """
    create a new directory with named date and time in "path" for saving data
    """
    if type=="now":
        today = datetime.datetime.now()
        todaystr = today.isoformat()
        os.makedirs(path+todaystr, exist_ok=True)
        return path+todaystr+"/"
    else:
        os.mkdir(path+type)
        return path+type+"/"

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as filename:
        return pickle.load(filename)

def save_dat(filename, data):
    file = open(filename,"w")
    file.write(data)
    file.close()