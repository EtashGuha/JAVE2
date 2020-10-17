class ArticleList():
	def __init__(self, model, topicfreq):
		self.articles = {}
		self.model = model
		self.topicfreq = topicfreq

	def add(article):
		modelScore = model.predict(article)
		freqScore = topicfreq.predict(article)
		upvotes = 0
		downvotes = 0
		socScore = socialScore(upvotes, downvotes)
		totalScore = (modelScore + freqScore + socScore)/3
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes)

	def socialScore(self, num_up, num_down):
		return (num_up + 1)/(num_down + 1)

	def upvote(article):
		totalScore, modelScore, freqScore, socScore, upvotes, downvotes = self.articles[article]
		upvotes += 1
		socScore = socialScore(upvotes, downvotes) 
		self.articles[article] = totalScore, modelScore, freqScore, socScore, upvotes, downvotes

	def downvote(article):
		totalScore, modelScore, freqScore, socScore, upvotes, downvotes = self.articles[article]
		downvotes += 1
		socScore = socialScore(upvotes, downvotes) 
		self.articles[article] = totalScore, modelScore, freqScore, socScore, upvotes, downvotes

	def getList():
		{k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
