class StepOne:
    def __init__(self, s1_model, params) -> None:
        self.model = s1_model
        self.params = params

    def training(self, **kwargs):
        self.model.load_data(kwargs['X_shared'], kwargs['Xs'])
        self.model.learning()
        Emb_U, Emb_US = self.model.extract_embedding()

        return Emb_U, Emb_US
