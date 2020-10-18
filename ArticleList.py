import math
import pickle
import os
import sys

class ArticleList():
	def __init__(self, model, topicfreq):
		self.articles = {}
		self.model = model
		self.topicfreq = topicfreq
		print(os.path.exists("important"))
		if os.path.exists("important"):
			file = open('important', 'rb')
			self.articles = pickle.load(file)
			file.close()


	def add(self, article, image, title, url):
		modelScore = self.model.predict(article)
		freqScore = modelScore
		upvotes = 0
		downvotes = 0
		socScore = self.socialScore(upvotes, downvotes)
		totalScore = (modelScore + freqScore + socScore)/3
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title)
		self.save()

	def socialScore(self, num_up, num_down):
		return math.atan((num_up + 1)/(num_down + 1)) * 2/math.pi

	def upvote(self, article):
		totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title = self.articles[article]
		upvotes += 1
		socScore = self.socialScore(upvotes, downvotes) 
		totalScore = (modelScore + freqScore + socScore)/3
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title)
		self.save()

	def downvote(self, article):
		totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title = self.articles[article]
		downvotes += 1
		socScore = self.socialScore(upvotes, downvotes) 
		totalScore = (modelScore + freqScore + socScore)/3
		self.articles[article] = (totalScore, modelScore, freqScore, socScore, upvotes, downvotes, url, image, title)
		self.save()

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

	def save(self):
		file = open('important', 'wb')

		# dump information to that file
		pickle.dump(self.articles, file)

		# close the file
		file.close()
