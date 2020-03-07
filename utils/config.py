import os
import yaml


def load_yml():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(dir_path, 'default_config.yml')
    with open(out_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yml():
    data = {'db_name': 'simpsons_7',
            'base_path': r'/home/frank/Documents/simpson_voices_7',
            'sound_device': 6,
            }

    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(dir_path, 'default_config.yml')
    with open(out_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    save_yml()
