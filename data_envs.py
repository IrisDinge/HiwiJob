import yaml


def initEnv(dataset_name):
    cfgs_root = 'cfgs'
    cur_cfg = getConfiguraton(cfgs_root)
    target = cur_cfg['dataset'][dataset_name]
    ret_cfg = combineConfig(target)
    return ret_cfg


def getConfiguraton(cfgs_root):
    main_cfg = parse('%s/configuration.yml' % cfgs_root)
    return main_cfg


def parse(fp):
    with open(fp, 'r') as fd:
        cont = fd.read()
        y = yaml.load(cont)
        return y


def combineConfig(target):
    ret_cfg = {}
    for k, v in target.items():
        ret_cfg[k] = v
    return ret_cfg


if __name__=='main':
    pass
