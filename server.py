import flask
from flask import request
from bert_model import BertModel
from ArticleList import ArticleList
from newspaper import Article
from flask import jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'


model = BertModel()
article_list = ArticleList(model, model)



@app.route('/', methods=['POST'])
def home():
	url = request.form['url']
	article = Article(url)
	article.download()
	article.parse()
	article_list.add(article.text, article.top_image, article.title, url)
	response = flask.jsonify({'some': 'data'})
	response.headers.add('Access-Control-Allow-Origin', '*')
	return response

@app.route('/', methods=['GET'])
def query():
	response = jsonify(article_list.getList())
	response.headers.add('Access-Control-Allow-Origin', '*')
	return response

@app.route('/upvote/', methods=['POST'])
def upvote():
	url = request.form['url']
	article = Article(url)
	article.download()
	article.parse()
	article_list.upvote(article.text)
	response = flask.jsonify({'some': 'data'})
	response.headers.add('Access-Control-Allow-Origin', '*')
	return response

@app.route('/downvote/', methods=['POST'])
def downvote():
	url = request.form['url']
	article = Article(url)
	article.download()
	article.parse()
	article_list.downvote(article.text)
	response = flask.jsonify({'some': 'data'})
	response.headers.add('Access-Control-Allow-Origin', '*')
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0')