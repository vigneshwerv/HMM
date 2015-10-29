from __future__ import division

class GramDict(object):

	def __init__(self):
		self.count = dict()
		self.dictionary = dict()

	def add(self, key, value):
		if key not in self.dictionary:
			self.count[key] = 0
			self.dictionary[key] = dict()
		if value not in self.dictionary[key]:
			self.dictionary[key][value] = 0
		self.dictionary[key][value] += 1
		self.count[key] += 1

	def normalize(self):
		for key_dict in self.dictionary.iterkeys():
			for value_dict in self.dictionary[key_dict].iterkeys():
				self.dictionary[key_dict][value_dict] /= self.count[key_dict]

	def get(self, key, value):
		if key in self.dictionary and value in self.dictionary[key]:
			return self.dictionary[key][value]
		return False

class HMM(object):

	def __init__(self):
		self.states = set()
		self.stateDictionary = GramDict()
		self.emissionDictionary = GramDict()
		self.initDictionary = GramDict()
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

				self.emissionDictionary.add(posTag, word)

				if prevPosTag == None:
					self.initDictionary.add('init', posTag)
					prevPosTag = posTag
					continue

				self.stateDictionary.add(prevPosTag, posTag)
				prevPosTag = posTag

		self.emissionDictionary.normalize()
		self.stateDictionary.normalize()
		self.initDictionary.normalize()

		self.states.update(self.stateDictionary.dictionary.keys())

	def __initial_probability__(self, state):
		return self.initDictionary.get('init', state) or 0

	def __emission_probability__(self, state, observation):
		return self.emissionDictionary.get(state, observation) or 0.0000015

	def __transition_probability__(self, prevState, nextState):
		return self.stateDictionary.get(prevState, nextState) or 0.0000015

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
