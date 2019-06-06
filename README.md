# htwdd-keras-text-summarization

test.py 
  -> Laden und Aufbereiten der Trainingsdaten 
  -> data/hackernoontutorial/Reviews.csv -> https://www.kaggle.com/snap/amazon-fine-food-reviews/

data_generator.py
  -> Generator-Klasse

seq2seq.py 
  -> Training mit aufbereiteten Trainingsdaten
  -> data/hackernoontutorial/reviews_dataset.pkl

seq2seq_predict.py 
  -> Prediction (noch mit Trainingsdaten)
