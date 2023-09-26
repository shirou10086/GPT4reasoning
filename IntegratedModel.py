class IntegratedModel(nn.Module):
    def __init__(self, simclr_model, text_image_model):
        super(IntegratedModel, self).__init__()
        self.simclr_model = simclr_model
        self.text_image_model = text_image_model

    def forward(self, image, text):
        image_feature = self.simclr_model(image)
        text_feature = self.text_image_model(text)
        integrated_feature = torch.cat((image_feature, text_feature), dim=-1)
        return integrated_feature
