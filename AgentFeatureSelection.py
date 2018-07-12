
#import SpamDetector as SD
import PhishingDetector as PD
import numpy as np
import random

feature_size_phishing = 30
feature_size_spam = 141

class Agent():

	# Init an Agent
	def __init__(self, features_size):

		"""
			Start a random list of 0 and 1 of len = features_size
			e.g. if features_size = 5 the chromosome = [1, 1, 1, 0, 1]
			features_size represent the amount of features
		"""
		self.chromosome = []
		for x in range(features_size):
			self.chromosome.append(random.randint(0,1))

		self.chromosome = np.array(self.chromosome)
		self.fitness = -1


	def __str__(self):
		return "Chromosome: " + str(self.chromosome) + ", with fitness " + str(self.fitness)


	features_size = 30
	population = 20
	generations = 100

	def ga():

		agents = Agent.init_agents(Agent.population, Agent.features_size)

		for generation in range(Agent.generations):

			print("Generation: "+str(generation))

			agents = Agent.fitness(agents)
			agents = Agent.selection(agents)
			agents = Agent.crossover(agents)
			agents = Agent.mutation(agents)


			if any(agent.fitness >= 0.9 for agent in agents):

				print("Found an agent")
				exit(0)


	# This function creates initial population using the Agent class, the return is a list
	# size population and each agent in the population must be size features_size
	def init_agents(population, features_size):

		return [Agent(features_size) for _ in range(population)]

	# This function will calculate the fitness in each memeber of the population
	def fitness(agents):

		for agent in agents:
			# Generate a phishing_detector for each agent
			pd = PD.phishing_detector(agent.chromosome)
			agent.fitness = pd.test_features_svm()
			#print(agent)

		return agents

	# The selection will select the population to be go for the next generation, 
	# the population will be decide by the highest fitness function higher the 
	# probability to be selected
	def selection(agents):

		agents = sorted(agents, key = lambda agent: agent.fitness, reverse = True)
		print('\n'.join(map(str, agents)))
		agents = agents[:int(0.3 * len(agents))]
		return agents

	# The crossover will combine the agents that were selected in the selection function
	def crossover(agents):
		new_blood = []
		for _ in range(int((Agent.population - len(agents))/ 2)):
			parent1 = random.choice(agents)
			parent2 = random.choice(agents)
			child1 = Agent(feature_size_phishing)
			child2 = Agent(feature_size_phishing)
			split_point = random.randint(0, feature_size_phishing)
			child1.chromosome = np.concatenate((parent1.chromosome[0:split_point], parent2.chromosome[split_point:feature_size_phishing]))
			child2.chromosome = np.concatenate((parent2.chromosome[0:split_point], parent1.chromosome[split_point:feature_size_phishing]))
			new_blood.append(child1)
			new_blood.append(child2)

		agents.extend(new_blood)

		return agents

	# The mutation will do random modification of the agents
	def mutation(agents):
		for agent in agents:
			for idx, param in enumerate(agent.chromosome):
				if random.uniform(0.0, 1.0) <= 0.05:
					if agent.chromosome[idx] == 1:
						new_value = np.array([0])
					else:
						new_value = np.array([1])
					agent.chromosome = np.concatenate((agent.chromosome[0:idx], new_value, agent.chromosome[idx+1:feature_size_phishing]))
		return agents



if __name__ == '__main__':
	"""
	agent1 = Agent(30)
	agent2 = Agent(30)
	agent3 = Agent(30)
	agent4 = Agent(30)
	agent5 = Agent(30)
	agent6 = Agent(30)

	agent1.chromosome =  np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
	agent2.chromosome =  np.array([1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
	agent3.chromosome =  np.array([1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.])
	agent4.chromosome =  np.array([1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
	agent5.chromosome =  np.array([1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
	agent6.chromosome =  np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

	agents = [agent1, agent2, agent3, agent4, agent5, agent6]

	Agent.fitness(agents)
	print('-----------------------------Fitness-------------------------------')
	print('\n'.join(map(str, agents)))

	Agent.selection(agents)
	print('-----------------------------Selection-------------------------------')
	print('\n'.join(map(str, agents)))

	Agent.crossover(agents)
	print('-----------------------------Crossover-------------------------------')
	print('\n'.join(map(str, agents)))
	
	Agent.mutation(agents)
	print('-----------------------------Mutation-------------------------------')
	print('\n'.join(map(str, agents)))

	"""
	Agent.ga()

	## Run the code for the selection



