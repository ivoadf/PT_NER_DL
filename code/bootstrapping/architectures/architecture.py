class Architecture:

    def train(self):
        raise NotImplementedError("Method not implemented")

    def test(self):
        raise NotImplementedError("Method not implemented")

    def name_list(self):
        raise NotImplementedError("Method not implemented")

    def set_test_dataset(self):
        raise NotImplementedError("Method not implemented")

    def set_train_dataset(self):
        raise NotImplementedError("Method not implemented")

    def set_embeddings(self):
        raise NotImplementedError("Method not implemented")

    def save_model(self):
        raise NotImplementedError("Method not implemented")
