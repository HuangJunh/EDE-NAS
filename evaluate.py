from utils import Utils, Log, GPUTools
from multiprocessing import Process, Manager
import importlib
import sys,os, time
import numpy as np
import copy
from asyncio.tasks import sleep


def decode(individual, curr_gen, id):

    pytorch_filename = Utils.generate_pytorch_file(individual, curr_gen, id)

    return pytorch_filename

def check_all_finished(filenames, curr_gen):
    filenames_ = copy.deepcopy(filenames)
    output_file = './populations/after_%02d.txt' % (curr_gen)
    assert os.path.exists(output_file) == True
    f = open(output_file, 'r')
    for line in f:
        if len(line.strip()) > 0:
            line = line.strip().split('=')
            if line[0] in filenames_:
                filenames_.remove(line[0])
    f.close()
    if filenames_:
        return False
    else:
        return True

def fitnessEvaluate(filenames, curr_gen, is_test):
    acc_set = np.zeros(len(filenames))
    params_set = np.zeros(len(filenames))
    has_evaluated_offspring = False
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    p = None
    for file_name in filenames:
        has_evaluated_offspring = True
        # time.sleep(20)
        if p:
            p.join()
        gpu_id = GPUTools.detect_available_gpu_id()
        while gpu_id is None:
            time.sleep(120)
            gpu_id = GPUTools.detect_available_gpu_id()
        if gpu_id is not None:
            Log.info('Begin to train %s' % (file_name))
            module_name = 'scripts.%s' % (file_name)
            if module_name in sys.modules.keys():
                Log.info('Module:%s has been loaded, delete it' % (module_name))
                del sys.modules[module_name]
                _module = importlib.import_module('.', module_name)
            else:
                _module = importlib.import_module('.', module_name)
            _class = getattr(_module, 'RunModel')
            cls_obj = _class()
            p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), curr_gen, file_name, is_test, return_dict))
            jobs.append(p)
            p.start()

    p.join()
    time.sleep(10)

    if has_evaluated_offspring:
        file_name = './populations/after_%02d.txt' % (curr_gen)
        assert os.path.exists(file_name) == True
        f = open(file_name, 'r')
        fitness_map = {}
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')
                fitness_map[line[0]] = float(line[1])
        f.close()

        for i in range(len(acc_set)):
            if filenames[i] not in fitness_map:
                Log.warn('The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 60 seconds' % (filenames[i], file_name))
                sleep(120)
            acc_set[i] = fitness_map[filenames[i]]
        #############################################################
        file_name = './populations/params_%02d.txt' % (curr_gen)
        assert os.path.exists(file_name) == True
        f = open(file_name, 'r')
        fitness_map = {}
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')
                fitness_map[line[0]] = float(line[1])
        f.close()

        for i in range(len(params_set)):
            if filenames[i] not in fitness_map:
                Log.warn(
                    'The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 60 seconds' % (
                    filenames[i], file_name))
                sleep(120)
            params_set[i] = fitness_map[filenames[i]]

    else:
        Log.info('None offspring has been evaluated')

    return list(acc_set), list(params_set)



