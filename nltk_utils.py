import nltk


class NLTK_UTILS:

  def __init__(self):
    pass
  
  def download(self) -> None:   
      nltk.download('punkt')
      nltk.download('stopwords')
      nltk.download('averaged_perceptron_tagger')
      return None
