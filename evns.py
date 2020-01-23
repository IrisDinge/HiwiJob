import yaml
import os
import sys


def initEnv(model_name, train_flag):
    cfgs_root = 'cfgs'
    cur_cfg = getConfig(cfgs_root, model_name)

    root_dir = cur_cfg['output_root']
    cur_cfg['model_name'] = model_name
    work_dir = os.path.join(root_dir, model_name)

    backup_name = cur_cfg['backup_name']
    log_name = cur_cfg['log_name']

    log_dir = os.path.join(work_dir, log_name)
    backup_dir = os.path.join(work_dir, backup_name)

    gpus = cur_cfg['train']['gpus']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    ret_cfg = combineConfig(cur_cfg, train_flag)

    return ret_cfg

def safeMakeDirs(tdir):
    if not os.path.isdir(tdir):
        os.mknod(tdir)


def getConfig(cfgs_root, model_name):
    main_cfg = parse('%s/main.yml' % cfgs_root)
    if model_name not in main_cfg['cfg_dict'].keys():
        models = ', '.join(main_cfg['cfg_dict'].keys())
        print('There are models like %s\n' % models, file=sys.stderr)
        raise Exception
    cfg_fp = './' + cfgs_root + '/' + main_cfg['cfg_dict'][model_name]
    #print(cfg_fp)
    config = parse(cfg_fp)
    return config


def parse(fp):
    with open(fp, 'r') as fd:
        cont = fd.read()
        y = yaml.load(cont)
        return y


def combineConfig(cur_fig, train_flag):
    ret_cfg = {}
    for k, v in cur_fig.items():
        if k == 'train' or k == 'test':
            continue
        ret_cfg[k] =v
    if train_flag == 1:
        key = 'train'
    if train_flag ==2:
        key = 'test'

    for k,v in cur_fig[key].items():
        ret_cfg[k] = v
    return ret_cfg





