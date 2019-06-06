import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_token_index, target_token_index,
                 input_texts, target_texts,
                 max_encoder_seq_length, num_encoder_tokens, max_decoder_seq_length, num_decoder_tokens,
                 batch_size=32, n_channels=1, shuffle=True):
        'Initialization'
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.max_encoder_seq_length = max_encoder_seq_length
        self.num_encoder_tokens = num_encoder_tokens
        self.max_decoder_seq_length = max_decoder_seq_length
        self.num_decoder_tokens = num_decoder_tokens
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input_texts) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        input_texts_temp = [self.input_texts[k] for k in indexes]
        target_texts_temp = [self.target_texts[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(input_texts_temp, target_texts_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.input_texts))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, input_texts, target_texts):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        encoder_input_data = np.zeros(
            (self.batch_size, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (self.batch_size, self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (self.batch_size, self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        # Generate data
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.

        return [encoder_input_data, decoder_input_data], decoder_target_data
