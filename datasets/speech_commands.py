import os

from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, set_name):
        super().__init__('./', download=True)

        def load_list(list_name):
            list_path = os.path.join(self._path, list_name)
            with open(list_path) as list_obj:
                return [os.path.join(self._path, file_path.strip()) for file_path in list_obj]

        testing_list_name = 'testing_list.txt'
        validation_list_name = 'validation_list.txt'
        if set_name == 'testing':
            self._walker = load_list(testing_list_name)
        elif set_name == 'validation':
            self._walker = load_list(validation_list_name)
        elif set_name == 'training':
            excludes = set(load_list(testing_list_name) + load_list(validation_list_name))
            self._walker = [file_path for file_path in self._walker if file_path not in excludes]
