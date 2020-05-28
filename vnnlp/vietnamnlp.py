import numpy as np
import pandas as pd
from typing import List
from vncorenlp import VnCoreNLP

# def vn_tokenize(csv_filepath: str) -> None:
def vn_tokenize(df: pd.DataFrame) -> pd.DataFrame:
    def _cut_dimension(element: List[str]) -> List[str]:
        dummy = []
        for elem in element:
            for ele in elem:
                dummy.append(ele)
        return dummy

    # df = pd.read_csv(csv_filepath)
    df['tokenized-text'] = pd.Series([], dtype=np.unicode_)
    buffr = []

    annotator = VnCoreNLP('./VnCoreNLP-1.1.1.jar', annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')

    for text in df['text']:
        word_segmented_text = annotator.tokenize(text)
        word_segmented_text = _cut_dimension(word_segmented_text)
        buffr.append(word_segmented_text)

    df['tokenized-text'] = buffr
    # np.save('tokenized-news.npy', buffr)
    return df



