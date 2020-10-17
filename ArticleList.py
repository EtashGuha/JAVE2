class ArticleList():
	def __init__(self, model, topicfreq):
		self.articles = {}
		self.model = model
		self.topicfreq = topicfreq

	def add(self, article, image, title, url):
		modelScore = self.model.predict(article)
		freqScore = self.topicfreq.predict(article)
		upvotes = 0
		downvotes = 0
		socScore = self.socialScore(upvotes, downvotes)
		totalScore = (modelScore + freqScore + socScore)/3
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title)
		return
	def socialScore(self, num_up, num_down):
		return (num_up + 1)/(num_down + 1)

	def upvote(self, article):
		totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title = self.articles[article]
		upvotes += 1
		socScore = self.socialScore(upvotes, downvotes) 
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title)

	def downvote(self, article):
		totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title = self.articles[article]
		downvotes += 1
		socScore = self.socialScore(upvotes, downvotes) 
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title)

	def getList(self):
		output = list(self.articles.items())
		output.sort(key = lambda x: x[1][0])  
		return list(map(self.label, output))

	def label(self, output):
		final = {}
		final["text"] = output[0]
		final["totalScore"] = output[1][0]
		final["modelScore"] = output[1][1]
		final["freqScore"] = output[1][2]
		final["socScore"] = output[1][3]
		final["upvotes"] = output[1][4]
		final["downvotes"] = output[1][5]
		final["url"] = output[1][6]
		final["image"] = output[1][7]
		final["title"] = output[1][8]
		return final