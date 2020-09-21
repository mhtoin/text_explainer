from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
import eli5
from eli5.lime import TextExplainer
from IPython.core.display import display, HTML
from eli5 import explain_weights, explain_prediction
from eli5.formatters import format_as_html, format_as_text, format_html_styles, fields
import webbrowser

# https://stackoverflow.com/questions/54908239/how-can-i-display-a-ipython-core-display-html-object-in-spyder-ipython-console

def print_prediction(doc):
    y_pred = pipe.predict_proba([doc])[0]
    for target, prob in zip(twenty_train.target_names, y_pred):
        print("{:.3f} {}".format(prob, target))    

#categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

twenty_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers'),
)
twenty_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers'),
)

fetch_subset = lambda subset: fetch_20newsgroups(
    subset=subset, categories=categories,
    shuffle=True, random_state=42,
    remove=('headers', 'footers', 'quotes'))
train = fetch_subset('train')
test = fetch_subset('test')


vec = TfidfVectorizer(min_df=3, stop_words='english',
                      ngram_range=(1, 2))
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
lsa = make_pipeline(vec, svd)

clf = SVC(C=150, gamma=2e-2, probability=True)
pipe = make_pipeline(lsa, clf)
pipe.fit(twenty_train.data, twenty_train.target)
pipe.score(twenty_test.data, twenty_test.target)

doc = twenty_test.data[0]
print_prediction(doc)

te = TextExplainer(random_state=42)
te.fit(doc, pipe.predict_proba)
#print(te.explain_prediction(target_names=twenty_train.target_names))
#print(eli5.format_as_image(te.explain_weights(target_names=twenty_train.target_names)))

show_html = lambda html: display(HTML(html))
show_html_expl = lambda expl, **kwargs: show_html(format_as_html(expl, include_styles=False, **kwargs))
show_html(format_html_styles())

weights = eli5.show_weights(clf, vec=vec, target_names=train['target_names'], horizontal_layout=False)

pred = show_html_expl(explain_prediction(clf, test['data'][2], vec, target_names=train['target_names']), force_weights=False, horizontal_layout=True)

with open('weights.htm', 'wb') as f:
    f.write(weights.data.encode("UTF-8"))

url = r'weights.htm'
webbrowser.open(url, new=2)