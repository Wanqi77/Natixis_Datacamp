import torch
import torch.nn as nn
import transformers

# Encoder: using pretrained model
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_name, dropout=0.5, fine_tune=True):
        super(Encoder, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        #self.output_size = output_size

        if fine_tune:
            self.fine_tune()

    def forward(self, speech_text):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.transformer(speech_text)  # (batch_size, transformer_output_size)
        out = out.last_hidden_state.mean(1)
        #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.transformer.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.transformer.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

# self defined Deocoder
class Decoder_Classif(nn.Module):
    def __init__(self, stock_data_size, input_size, encoded_size, hidden_size, dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        """
        super().__init__()
        self.input_size = input_size
        self.encoded_size = encoded_size 
        self.stock_data_size = stock_data_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.encoder = torch.nn.Linear(input_size, encoded_size)

        self.network = torch.nn.Sequential(
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(encoded_size + stock_data_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1),
            )
        self.sigmoid = torch.nn.Sigmoid()
        #self.init_weights()

    def forward(self, input, stock_price):
        """
        Forward propagation.

        :param input: encoded speech text, the output of the Encoder
        :param stock_price: the stock price        
        :return: classification prob
        """

        encoded_output = self.encoder(input)
        merged_data = torch.cat([encoded_output, stock_price],dim=1)
        out = self.network(merged_data)
        p = self.sigmoid(out)

        return p

