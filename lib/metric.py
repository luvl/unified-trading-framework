from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def roc_auc(y_label, y_predict, average='macro'):
    lb = LabelBinarizer()
    lb.fit(y_label)
    y_label = lb.transform(y_label)
    y_predict = lb.transform(y_predict)
    return roc_auc_score(y_label, y_predict, average=average)

