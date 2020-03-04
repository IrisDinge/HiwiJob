import sys
from pathlib import Path
import yaml
import tasks
import types


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Insufficient number of parameters, please specify a configuration file")
        sys.exit(-1)

    base_path = Path(__file__).absolute().parent
    config_file = base_path.joinpath('cfgs', sys.argv[1] + '.yml')


    if not config_file.is_file():
        print("Unknown configuration file")
        sys.exit(-2)

    def parse(d):
        result = types.SimpleNamespace()
        for k, v in d.items():
            if type(v) == dict:
                v = parse(v)
            setattr(result, k, v)
        return result

    with config_file.open() as f:
        print("Reading configuration file", config_file)
        cont = f.read()
        config = parse(yaml.load(cont, yaml.FullLoader))
        config_ymal = yaml.load(cont, yaml.FullLoader)

    if config.task == "dataprocessing":

        task = getattr(tasks, config.task)(config)
        task.run()
    else:
        action = getattr(tasks, config.task)
        task = getattr(action, config.task)(config)
        task.run()





