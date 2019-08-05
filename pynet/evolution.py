#--start requirements--
#pip installs
import numpy as np

#customs
import numpy_ext as npe
#from dictableobj import UniqueStatusObject

#builtins
import copy

#--end requirements--


class Population:#(UniqueStatusObject):
	"""Population of dna.
	Attributes:
		ID (str): Unique Identifier of the population
		census (list): List of integer dnas.
		seeds (list): List of integer dnas to seed this population.
		fitness (list): List of assessed fitness of individuals in population. Fitness between [0.0,1.0]
		parents (dict): parents[child] = [[mom_dna,dad_dna],...]
		generated (list): List of bool to denote which were generated randomly
		elite_percentage (float): Percentage of population considered to be elite and will automatically be selected to survive.
		survivor_percentage (float): Percentage of population that will survive.
		mutation_chance (float): Percent change of an allele to mutate (switch) during crossover.
	"""

	def __init__(self,census=[],seeds=None,fitness=[],parents={},generated=[],elite_percentage=0.1,survivor_percentage=0.1,mutation_chance=0.0001,**kwargs):

		self.census = census if census != [] else []
		self.seeds = seeds if seeds != None else None
		self.fitness = fitness if fitness != [] else []
		self.parents = parents if parents != {} else {}
		self.generated = generated if generated != [] else []
		self.elite_percentage = elite_percentage if elite_percentage != 0.1 else 0.1
		self.survivor_percentage = survivor_percentage if survivor_percentage != 0.1 else 0.1
		self.mutation_chance = mutation_chance if mutation_chance != 0.0001 else 0.0001

		return super().__init__(kwargs)

class Evolution:
	"""Class for optimizing problem space hyper parameters using a genetic method designed to mimic natural evolution.
	Notes:
		Usage Pattern:
		1. Parameterize a system and convert parameters to single dna.
		2. Establish performance evaluation (fitness) function to optimize.
		3. Create list of dnas and loop for generations.
		4. Each generation test the fitness of a dna.
		5. Use Evolution to select members for the next generation of a population.
	TODO:
		pop capacity not fixed for finite resources (like computation time for fitness func?)
		Add different crossover techniques
	"""

	def Selection(population: Population,total_dna_bits: int,survivor_percentage: float,elite_percentage: float,mutation_chance: float):
		"""Select and create the next generation of a population.
		Args:
			population (Population):
			total_dna_bits (int): Total bits the dna will need to be represented.			
			survivor_percentage (float): Percentage of population that will survive.
			elite_percentage (float): Percentage of population considered to be elite and will automatically be selected to survive.
			mutation_chance (float): Percent change of an allele to mutate (switch) during crossover.
		Returns:
			Population: next_gen
		Notes:
			1. The total percentage of the population that survives will be greater than the specified survivor_percentage if the elite_percentage is above 0.
			2. Breeding will continue until the population size is the same/larger than before.
			3. Individual cannot breed with itself.
			4. Fitness must be in range [0.0,1.0]
		"""

		fitness = np.array(population.fitness,np.float32)

		#rank with highest fitness at top
		ranks = np.argsort(1-fitness)

		#mark tops as elite, will survive no matter what
		elite_idx = int(np.ceil(elite_percentage * len(fitness)))
		elites = ranks[0:elite_idx]
		
		#randomly select non elites to survive (to prevent converging diversity)
		ranks_without_elites = ranks[elite_idx:]
		survivor_count  = int(np.ceil(survivor_percentage * len(ranks_without_elites)))
		survivors = ranks_without_elites[np.random.randint(len(ranks_without_elites),size=survivor_count)]

		#create next gen from elites and survivors
		survivors = np.concatenate((elites,survivors))
		population_next = np.take(population.census, survivors).tolist()

		#record how the new generation was created
		generated = []
		parents = copy.deepcopy(population.parents)
		for i in range(len(population_next)):
			generated.append(population.generated[survivors[i]]) #generated None means was a survivor
			#parents.append(population.parents[survivors[i]]) #dont record parents if survived?

		#apply percentage to be parent
		parent_likelihood = (len(fitness) - ranks) / np.sum(ranks)  #+1 because rank 0 division
		parent_likelihood = parent_likelihood/np.sum(parent_likelihood)
		parent_likelihood = np.cumsum(parent_likelihood)
		
		#breed until next generation is as large as previous
		while len(population_next) < len(population.census):
			mom = -1
			dad = -1
			while (dad == -1) or (mom == -1):
			#while dad is mom:
				#randomly select 2 parents
				parent_rand = np.random.rand(2)
				mom = ranks[parent_likelihood >= parent_rand[0]][0]
				dad = ranks[parent_likelihood >= parent_rand[1]][0]

			#have parents breed
			child = Evolution.Breed(population.census[mom],population.census[dad],mutation_chance,total_dna_bits)

			#parents = parents + population.parents
			try:
				parents[str(child)]
				parents[str(child)].append([population.census[mom],population.census[dad]])
			except:
				parents[str(child)] = [[population.census[mom],population.census[dad]]]

			#record a not generated child
			generated.append(False)
			population_next.append(child)

		return Population(**{
			'census': population_next,
			'fitness': [],
			'seeds': [],
			'generated': generated,
			'parents': parents,
			'mutation_chance': mutation_chance,
			'elite_percentage': elite_percentage,
			'survivor_percentage': survivor_percentage,
			})

	def Breed(mom_dna: int,dad_dna: int,mutate_chance: float,total_dna_bits: int):
		"""Produce child dna from 2 parent dna through uniform crossover and bit inversion mutation.
		Args:
			mom_dna (int): DNA of parent 1
			dad_dna (int): DNA of parent 2
			mutate_chance (float): Chance of individual bit flipping.
			total_dna_bits (int): Total number of bits in DNA being bred.
		Returns:
			child_dna: int
		Ref:
			http://www.obitko.com/tutorials/genetic-algorithms/crossover-mutation.php
		TODO:
			Does rounding for use_moms affect the uniform nature?
			Change the bit inversion to be more efficient -> https://stackoverflow.com/questions/1228299/change-one-character-in-a-string/22149018#22149018
		"""

		# turn int to binary str -> list?
		mom_genes = list(npe.Int2Bin(mom_dna,total_dna_bits)[::-1])
		dad_genes = list(npe.Int2Bin(dad_dna,total_dna_bits)[::-1])

		#generate uniform random indices for using mom dna
		use_moms = np.round(np.random.rand(total_dna_bits)) #round to make int

		#generate uniform random chances of mutation
		mutate_rand = np.random.rand(total_dna_bits)

		#crossover
		child_genes = dad_genes #copy dad genes entirely as child
		for base_pair in range(0,len(mom_genes)):

			#uniform
			if bool(use_moms[base_pair]): #if use mom allele, replace copy of dads with mom
				child_genes[base_pair] = mom_genes[base_pair]

			#invert bit for mutation
			if mutate_rand[base_pair] <= mutate_chance: #if mutate, flip the bit
				if child_genes[base_pair] is '0':
					child_genes[base_pair] = '1'
				else:
					child_genes[base_pair] = '0'

		#join the list and turn into dna int
		child_dna = npe.Bin2Int(''.join(child_genes))

		return child_dna

class Tests:

	def main():

		Tests.test_evolving()

		pass

	def test_evolving():

		#TODO:
		#essentially create a census of dnas
		#then a simple perhaps random fitness fcn
		#select/breed and produce next gen

		dad_dna = 124
		mom_dna = 200
		n_bits = 8
		
		mutate = 0.1

		children = []
		for _ in range(100):
			child_dna = Evolution.Breed(mom_dna,dad_dna,mutate_chance=mutate,total_dna_bits=n_bits)
			children.append(child_dna)


		pass