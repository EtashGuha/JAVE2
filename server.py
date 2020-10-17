import flask
from flask import request
from bert_model import BertModel
from ArticleList import ArticleList

app = flask.Flask(__name__)
app.config["DEBUG"] = True

model = BertModel()
article_list = ArticleList(model, model)


@app.route('/', methods=['POST'])
def home():
	url = request.form['url']
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


app.run()