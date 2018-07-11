
class Agent():

	# Init an Agent
	def __init__(self, features_size):

		"""
			Start a random list of 0 and 1 of len = features_size
			e.g. if features_size = 5 the chromosome = [1, 1, 1, 0, 1]
			features_size represent the amount of features
		"""
		self.chromosome = []
		self.fitness = -1


	def __str__(self):
		return "Chromosome: " + str(self.chromosome) + ", with fitness " + self.fitness


	features_size = None
	population = 20
	generations = 100

	def ga():

		agents = init_agents(population, features_size)

		for generation in range(generations):

			print("Generation: "+str(generation))

			agents = fitness(agents)
			agents = selection(agents)
			agents = crossover(agents)
			agents = mutation(agents)


			if any(agent.fitness >= 0.9 for agent in agents):

				print("Found an agent")
				exit(0)


	# This function creates initial population using the Agent class, the return is a list
	# size population and each agent in the population must be size features_size
	def init_agents(population, features_size):

		return agents

	# This function will calculate the fitness in each memeber of the population
	def fitness(agents):

		return agents

	# The selection will select the population to be go for the next generation, 
	# the population will be decide by the highest fitness function higher the 
	# probability to be selected
	def selection(agents):

		return agents

	# The crossover will combine the agents that were selected in the selection function
	def crossover(agents):

		return agents

	# The mutation will do random modification of the agents
	def mutation(agents):

		return agents



if __name__ == '__main__':

	## Run the code for the selection



