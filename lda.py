
#Libraries for feature extraction and topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.sklearn
import mglearn
#Other libraries
import numpy as np
import pandas as pd



financedoc = "ssdsdksdksdk"

vect=CountVectorizer(ngram_range=(1,1),stop_words='english')
fin=vect.fit_transform(financedoc)
# pd.DataFrame(fin.toarray(),columns=vect.get_feature_names()).head(1)


lda=LatentDirichletAllocation(n_components=5)
lda.fit_transform(fin)
lda_dtf=lda.fit_transform(fin)

# sorting=np.argsort(lda.components_)[:,::-1]
# features=np.array(vect.get_feature_names())

# array=np.full((1, sorting.shape[1]), 1)
# array = np.concatenate((array,sorting), axis=0)

# topics = mglearn.tools.print_topics(topics=range(1,6), feature_names=features,
# sorting=array, topics_per_chunk=5, n_words=10)

zit=pyLDAvis.sklearn.prepare(lda,fin,vect)
pyLDAvis.display(zit)