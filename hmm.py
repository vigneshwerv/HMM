from __future__ import division

class HMM(object):

	def __init__(self):
		self.states = set()
		self.stateDictionary = dict()
		self.emissionDictionary = dict()
		self.initProbability = {'count': 0}
		pass

	def trainHMM(self, filename):

		prevPosTag = None
		with open(filename, "r") as train_file:

			for line in train_file:

				line = line.strip()
				if line == "###/###":
					prevPosTag = None
					continue

				word, posTag = line.split('/')
				posTag = posTag.replace('\n', '')
				word = word.lower()

				if posTag not in self.emissionDictionary:
					self.emissionDictionary[posTag] = {'count': 0, 'emis': dict()}
					self.stateDictionary[posTag] = {'count': 0, 'next': dict()}
				if word not in self.emissionDictionary[posTag]['emis']:
					self.emissionDictionary[posTag]['emis'][word] = 0

				self.emissionDictionary[posTag]['count'] += 1
				self.emissionDictionary[posTag]['emis'][word] += 1

				if prevPosTag == None:
					if posTag not in self.initProbability:
						self.initProbability[posTag] = 0
					self.initProbability[posTag] += 1
					self.initProbability['count'] += 1
					prevPosTag = posTag
					continue

				self.stateDictionary[prevPosTag]['count'] += 1
				if posTag not in self.stateDictionary[prevPosTag]['next']:
					self.stateDictionary[prevPosTag]['next'][posTag] = 0
				self.stateDictionary[prevPosTag]['next'][posTag] += 1
				prevPosTag = posTag

		for posTag in self.stateDictionary.iterkeys():
			self.states.update(posTag)
			count = self.stateDictionary[posTag]['count']
			for nextPosTags in self.stateDictionary[posTag]['next'].iterkeys():
				self.stateDictionary[posTag]['next'][nextPosTags] /= count
			count = self.emissionDictionary[posTag]['count']
			for emission in self.emissionDictionary[posTag]['emis'].iterkeys():
				self.emissionDictionary[posTag]['emis'][emission] /= count

		count = self.initProbability['count']
		for posTag in self.initProbability.iterkeys():
			if posTag == 'count':
				continue
			self.initProbability[posTag] /= count

	def __initial_probability__(self, state):
		return self.initProbability[state] if state in self.initProbability else 0

	def __emission_probability__(self, state, observation):
		if observation not in self.emissionDictionary[state]['emis']:
			return 0.0000015
		return self.emissionDictionary[state]['emis'][observation]

	def __transition_probability__(self, prevState, nextState):
		if nextState not in self.stateDictionary[prevState]['next']:
			return 0.0000015
		return self.stateDictionary[prevState]['next'][nextState]

	def testHMM(self, filename):
		with open(filename, "r") as test_file:
			actual_count = 0
			successful_count = 0
			true_states = []
			obs = []
			for line in test_file:
				line = line.strip()
				if line != '###/###':
					word, tag = line.split('/')
					true_states.append(tag)
					obs.append(word.lower())
					continue
				if len(true_states) == 0:
					continue
				(probability, pred_states) = self.__predict_tagsets__(obs)
				for i in xrange(len(true_states)):
					if true_states[i] == pred_states[i]:
						successful_count += 1
					actual_count += 1
				#print pred_states
				#print true_states
				print (actual_count - successful_count)/actual_count * 100
				obs = []
				true_states = []
				#raw_input()

	def __predict_tagsets__(self, observations):
		viterbi = [{}]
		path = {}
	    
		for s0 in self.states:
			viterbi[0][s0] = self.__initial_probability__(s0) * self.__emission_probability__(s0, observations[0])
			path[s0] = [s0]

		for t in range(1, len(observations)):
			viterbi.append({})
			newpath = {}

			for sj in self.states:
				(prob, state) = max((viterbi[t-1][si] * self.__transition_probability__(si, sj) * self.__emission_probability__(sj, observations[t]), si) for si in self.states)
				viterbi[t][sj] = prob
				newpath[sj] = path[state] + [sj]
			path = newpath
		n = 0
		if len(observations) != 1:
			n = len(observations) - 1
		(prob, state) = max((viterbi[n][y], y) for y in self.states)
		return (prob, path[state])



if __name__ == '__main__':
	hmm = HMM()
	hmm.trainHMM("entrain.txt")
	hmm.testHMM("entest.txt")