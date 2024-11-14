from nlp4bia.datasets.Dataset import Dataset

class Distemist(Dataset):
    def __init__(self, path, lang="es"):
        self.path = path
        self.task_name = "multilingual"
        self.lang = None
        self.data = None
        self.preprocessed_data = None

    def load_data(self, indices=None):
        with open(self.path, "r") as f:
            self.data = f.readlines()
        if indices:
            self.data = [self.data[i] for i in indices]

    def preprocess_data(self):
        self.preprocessed_data = [x.strip() for x in self.data]

    def __repr__(self):
        return f"Dataset -- {self.path}, Task: {self.task_name}"