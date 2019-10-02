#--start requirements--
#pip installs
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import tensorflow_transform as tft
from evolution import Evolution, Population
import numpyextension as npe

#customs

#builtins
import json
import time
import copy
import os

#--end requirements--

class Pynet:
	"""Artificial neural network model based on matlab neural network toolbox.
	Args:
		dna (int):
		preProcessor (PreProcess): Preprocessing unit for morphing data to before entering the network.
		trainer (Training): Training unit for optimizing to a dataset.
		isInput (list): List of bools denoting whether corresponding layer is connected to the input.
		isOutput (list): List of bools denoting whether corresponding layer is connected to the output.
		layerConnect (list): Jagged list of bools denoting which layer takes input from others.
		layers (dict): Key is layer index, value is the Layer itself.
		n_features (int): Count of individual features/measurements of the input dataset.
		n_classes (int): Count of individual classifications/values of the output dataset.
		useGPU (bool): True to configure and utilize the GPU for computation.
	Notes:
		Uses 32-bit float for calculations.
		While the Pynet may be loaded as an object, ConfigureGraph() must be called before it may perform.
		Goal is pointless for some like cross_entropy as it depends on groupsize -> loss is for whole group
		To view graph -> >>tensorboard --logdir /recordPath/graphs
	Ref:
		https://stackoverflow.com/questions/44880246/how-to-run-tensorboard-automatically-when-training-my-model
		https://en.wikipedia.org/wiki/Hologenome_theory_of_evolution
	TODO:
		Add integration tools to deploy pynets to AWS/DO for training in cloud.
		Add more tensorboard tracking/visualizing -> https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
	"""

	def __init__(self,kwargs):

		#dna int which represents the structure of the pynet
		self.dna = None if 'dna' not in kwargs.keys() else kwargs['dna']
					
		#preprocessing data and training on data
		self.preProcessor = kwargs['preProcessor']
		self.trainer = kwargs['trainer']

		#layers
		self.isInput = kwargs['isInput']
		self.isOutput = kwargs['isOutput']
		self.layerConnect = kwargs['layerConnect']
		self.layers = kwargs['layers']

		#input/output sizes
		self.n_features = kwargs['n_features']
		self.n_classes = kwargs['n_classes']
			
		#execution options
		self.useGPU = kwargs['onGPU']
		self.isRecurrent = None

		#calculation holders, must configure first
		self.graph = None
		self.output = None
		self.update_wb = None
		self.error = None

		return super().__init__()

	def Dictify(self):
		"""Transform this pynet into a dict for saving.
		Returns:
			dict: net
		"""
		net = {}
		#extract the preprocessing settings
		processSettings = {
			'xmax': self.preProcessor.settings.xmax.tolist(),
			'xmin': self.preProcessor.settings.xmin.tolist(),
			'ymax': self.preProcessor.settings.ymax[0].tolist(), #TODO: if y is 1dim
			'ymin': self.preProcessor.settings.ymin[0].tolist(),
			'xmean': self.preProcessor.settings.xmean[0].tolist(),
			'xstd': self.preProcessor.settings.xstd[0].tolist()
			}
		inputs = []
		inputs.append({
			'processFcns': [self.preProcessor.processFcn],
			'processSettings': [processSettings],
			'size': self.n_features
			})

		net['inputs'] = inputs

		#extract params for each layer
		net['numLayers'] = len(self.layers)
		net['inputConnect'] = self.isInput
		net['outputConnect'] = self.isOutput
		net['layerConnect'] = self.layerConnect
			
		layers = []
		bs = []
		IWs = []
		LWs = []
		inputWeights = []
		layerWeights = []
		outputs = []
		for L in range(0,len(self.layers)):
			layer = {}
			layer['transferFcn'] = self.layers[L].transferFcn
			layer['netInputFcn'] = self.layers[L].inputFcn
			layer['initFcn'] = self.layers[L].initFcn
			layer['size'] = self.layers[L].n_nodes
			b = self.layers[L].bias
			IW = []
			inputWeight = []
			layerWeight = []
			output = []
			if self.isInput[L]:
				inputWeight = {}
				inputWeight['weightFcn'] = self.layers[L].weightFcn
				IW = self.layers[L].inputWeights.tolist()
			else:
				layerWeight = {}
				layerWeight['weightFcn'] = self.layers[L].weightFcn
				
			if self.isOutput[L]:
				output = {}
				output['size'] = self.n_classes

			layers.append(layer)
			bs.append(b.tolist())
			IWs.append(IW)
				
			inputWeights.append(inputWeight)
			layerWeights.append(layerWeight)
			outputs.append(output)
			for L_i in range(0,len(self.layers)):
				LWs.append([])
				
		li = 0
		for L_i in range(0,len(self.layers)):
			for L in range(0,len(self.layers)):
				if len(self.layers[L].layerWeights[L_i]) > 0:
					LW = self.layers[L].layerWeights[L_i].tolist()
					LWs[li] = LW
				li = li + 1

		net['layers'] = layers
		net['b'] = bs
		net['IW'] = IWs
		net['LW'] = LWs
		net['inputWeights'] = inputWeights
		net['layerWeights'] = layerWeights
		net['outputs'] = outputs

		#net['adaptFcn'] = self.adaptFcn
		net['dna'] = self.dna
			
		net['trainParam'] = {}
		net['performFcn'] = self.trainer.performFcn
		net['trainFcn'] = self.trainer.trainFcn
		net['trainParam']['showCommandLine'] = self.trainer.showCommandLine
		#net['trainParam']['show'] = self.trainer.show
		net['trainParam']['epochs'] = self.trainer.epochs
		#net['trainParam']['time'] = self.trainer.time
		net['trainParam']['goal'] = self.trainer.goal
		net['trainParam']['min_grad'] = self.trainer.min_grad
		net['trainParam']['max_fail'] = self.trainer.max_fail
		#net['trainParam']['sigma'] = self.trainer.sigma
		#net['trainParam']['lambda'] = self.trainer.lambd
		net['trainParam']['learning_rate'] = self.trainer.learning_rate
		net['divideFcn'] = self.trainer.divideFcn
		net['divideParam'] = self.trainer.divideParams

		return net

	def ConfigureGraph(self,showSteps=False):
		"""Configures the tensorflow graphs for this pynet. This should be run upon loading/creating a pynet.
		Args:
			showSteps (bool): True if to show steps of input signal through network via console printout.
		Notes:
			The first sim after a configure will take longer than the next ones due to memory initialization.
		"""

		if Pynet.Utils.IsRecurrent(self):
			#TODO: still working on recurrent models (time/sequence dependancy
			#https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f
			raise ValueError('Recurrent pynet still unspported :(')

		#create list of tensorflow ops for printing (should showSteps be True)
		print_ops = []

		#determine if this pynet is a group
		isGroup,sample_dim,n_members = Pynet.Utils.DetermineGroup(self)

		#assign processing unit
		if self.useGPU:
			device = '/device:GPU:0'
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
				
		else:
			device = '/device:CPU:0'
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #prevents allocating gpu memory
			
		#build graph
		g = tf.Graph()
		with g.as_default():
			with tf.device(device):

				#create network placeholders
				X_tf = tf.placeholder(tf.float32,name='X_tf')
				xmax_tf = tf.placeholder(tf.float32,name='xmax_tf')
				xmin_tf = tf.placeholder(tf.float32,name='xmin_tf')
				ymax_tf = tf.placeholder(tf.float32,name='ymax_tf')
				ymin_tf = tf.placeholder(tf.float32,name='ymin_tf')
				xmean_tf = tf.placeholder(tf.float32,name='xmean_tf')
				xstd_tf = tf.placeholder(tf.float32,name='xstd_tf')

				#create weights/bias tensorflow variables per layer
				iw,lw,b,var_list = Pynet.Utils.Build_Layer_tfVars(self,isGroup)

				#initialize layer_output_signals to zeros of correct size for feedback connections
				layer_output_tensors = Pynet.Utils.Initialize_layer_output_tensors(self,X_tf,isGroup,n_members,sample_dim)
					
				#preprocess input
				Xp_tf = PreProcess.DNA.fcns[self.preProcessor.processFcn](X_tf,xmax_tf,xmin_tf,ymax_tf,ymin_tf,xmean_tf,xstd_tf)
				#if showSteps: Xp_tf = tf.Print(Xp_tf, [tf.shape(Xp_tf),Xp_tf], message="--- Xp_tf0:\n")
				if showSteps: print_ops.append(tf.print('Xp_tf0: ',Xp_tf))
				
				#create layer dependancies to ensure proper calculation control/flow
				layer_input_dependancies,layer_output_dependancies = Pynet.Utils.Initialize_layer_dependancies(self,Xp_tf,b,iw,lw,layer_output_tensors)

				#propagate through layers
				for L in range(0, len(self.layers)):
					layer_inputs = []
					#with input control dependancies
					with tf.control_dependencies(layer_input_dependancies[L]):
						if self.isInput[L]: #if this layer takes input
							#weight function
							Z_tf = Layer.DNA.weight_fcns[self.layers[L].weightFcn](Xp_tf,iw[L])
							#if showSteps: Z_tf = tf.Print(Z_tf, [tf.shape(Z_tf),Z_tf], message="--- Z_tf_Li:" + str(L) + ":\n")
							if showSteps: print_ops.append(tf.print('Z_tf_Li:' + str(L) + ': ',Z_tf))
							layer_inputs.append(Z_tf) #add to list of total inputs to layer
								
						#collect layer input from other layer output
						for L_i in range(0, len(self.layers)):
							if self.layerConnect[L][L_i]: #if this layer takes layer input connection
								#weight function
								Z_tf = Layer.DNA.weight_fcns[self.layers[L].weightFcn](layer_output_tensors[L_i],lw[L][L_i])
								#if showSteps: Z_tf = tf.Print(Z_tf, [tf.shape(Z_tf),Z_tf], message="--- Z_tf_L:" + str(L) + ":\n")
								if showSteps: print_ops.append(tf.print('Z_tf_L:' + str(L) + ': ',Z_tf))
								layer_inputs.append(Z_tf)

					with tf.control_dependencies(layer_output_dependancies[L]): #with output control dependancies
						#bias function
						N_tf = Layer.DNA.input_fcns[self.layers[L].inputFcn](layer_inputs,b[L])
						#if showSteps: N_tf = tf.Print(N_tf, [tf.shape(N_tf),N_tf], message="--- N_tf_L:" + str(L) + ":\n")
						if showSteps: print_ops.append(tf.print('N_tf_L:' + str(L) + ': ',N_tf))

						#transfer function
						A_tf = Transfer.DNA.fcns[self.layers[L].transferFcn](N_tf,sample_dim + 1) #why +1?
						#if showSteps: A_tf = tf.Print(A_tf, [tf.shape(A_tf),A_tf], message="--- A_tf_L:" + str(L) + ":\n")
						if showSteps: print_ops.append(tf.print('A_tf_L:' + str(L) + ': ',A_tf))
							
						#TODO: still dont know how to hold onto layer output but also use an initialize zeros value
						#want to capture layer output after sim so if session is closed, can still use
						#this may also allow recurrent networks
						lastState_tf = tf.Variable(A_tf,validate_shape=False,name='A_tf' + str(L))

						layer_output_tensors[L] = A_tf #when tensor, not variable
						#overwrite the output tensor to be signal from layer
						#layer_output_tensors[L] = tf.assign(layer_output_tensors[L],A_tf)

						if self.isOutput[L]:
							#with tf.control_dependencies(layer_output_signal[L]):
							with tf.control_dependencies(print_ops):
								signal_tf = layer_output_tensors[L]

				#placeholder for targets
				T_tf = tf.placeholder(dtype=tf.float32,name='T_tf')

				#set performance function as a loss to minimize
				loss_tf = Performance.DNA.fcns[self.trainer.performFcn](T_tf,signal_tf)

				#use training function to update weights/bias
				update_wb = Training.DNA.fcns[self.trainer.trainFcn](self.trainer.min_grad,self.trainer.learning_rate,loss_tf,var_list)

		self.graph = g
		self.output = signal_tf
		self.update_wb = update_wb
		self.error = loss_tf

		return self

	def Loss(self,X: np.array,T: np.array,log_device_placement=False):
		"""Simulates an input signal to this neural network and returns the loss according to the performance function.
		Inputs:
			X (np.array): shape => [n_samples,n_features] or [n_members,n_samples,n_features]
			T (np.array): shape => [n_samples,n_classes] or [n_members,n_samples,n_classes]
			log_device_placement (bool): Print out  which devices your operations and tensors are assigned to.
		Returns:
			float: loss
		TODO:
			In configure change self.error so this result has a members dim?
		"""
		#determine if this is a group
		isGroup,sample_dim,n_members = Pynet.Utils.DetermineGroup(self)

		#feed dict start
		feed_dict = {
			'T_tf:0': T,
			'xmax_tf:0': self.preProcessor.settings.xmax,
			'xmin_tf:0': self.preProcessor.settings.xmin,
			'ymax_tf:0': self.preProcessor.settings.ymax,
			'ymin_tf:0': self.preProcessor.settings.ymin
			}

		#X must have a singleton dim
		if len(np.shape(X)) == 1:
			X = np.expand_dims(X,0)

		#build feed dict
		feed_dict,n_steps,step_dict = Pynet.Utils.X2feed_dict(self,X,isGroup,n_members,sample_dim,feed_dict)

		#start tensorflow session
		gpuOptions = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=True,gpu_options=gpuOptions)
		with tf.Session(graph=self.graph,config=config) as sess:
			
			#initialize variables
			sess.run(tf.global_variables_initializer(), feed_dict)

			#run the error of the pynet
			result = sess.run(self.error,feed_dict)
		
		return result

	def Sim(self,X: np.array,onehot=False,log_device_placement=False,recordGraph=False,recordPath='./Graphs/defaultGraphName'):
		"""Simulates an input signal to this pynet and returns the output signal
		Args:
			X (np.array): shape => [n_samples,n_features] or [n_members,n_samples,n_features]
			onehot (bool): Returns result with maximum signal as 1 and all others are 0
			log_device_placement (bool): Print out  which devices your operations and tensors are assigned to.
			recordGraph (bool): Whether to record the network using tensorboard.
			recordPath (str):  Path to record network graph to.
		Returns:
			np.array: shape => [n_samples,n_classes] or [n_members,n_samples,n_classes]
		TODO:
			Support recurrent/sequential networks
		"""
		#still not supporting recurrent
		if Pynet.Utils.IsRecurrent(self): raise ValueError('Recurrent pynet still unspported :(')

		#determine if this is a group
		isGroup,sample_dim,n_members = Pynet.Utils.DetermineGroup(self)

		#feed dict start
		feed_dict = {
			'xmax_tf:0': self.preProcessor.settings.xmax,
			'xmin_tf:0': self.preProcessor.settings.xmin,
			'ymax_tf:0': self.preProcessor.settings.ymax,
			'ymin_tf:0': self.preProcessor.settings.ymin,
			'xmean_tf:0': self.preProcessor.settings.xmean,
			'xstd_tf:0': self.preProcessor.settings.xstd,
			}

		#X must have a singleton dim
		if len(np.shape(X)) == 1: X = np.expand_dims(X,0)

		#build feed dict
		feed_dict,n_steps,step_dict = Pynet.Utils.X2feed_dict(self,X,isGroup,n_members,sample_dim,feed_dict)

		#start tensorflow session
		gpuOptions = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=True,gpu_options=gpuOptions)
		step = 0
		presult = {}
		with tf.Session(graph=self.graph,config=config) as sess:
			
			if recordGraph: writer = tf.summary.FileWriter(recordPath)

			#run the output of the pynet
			if self.isRecurrent:
				raise ValueError('not ready for recurrent yet - unfinished')
				#initialize variables
				feed_dict['X_tf:0'] = step_dict[0] #need shape of X for init
				sess.run(tf.global_variables_initializer(),feed_dict)

				for step in range(0,n_steps):
					feed_dict['X_tf:0'] = step_dict[step] #TODO can reassign a dict?
					presult[step] = sess.run(self.output, feed_dict)

				for L in range(0,len(self.layers)):
					#TODO: extract proper A_tf?  or just the initialized A_tf?  would naming
					#transfer op help?
					#wait, this isnt evena variable?
					self.layers[L].lastState = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'A_tf' + str(L))[0])
					a = 5

			else:
				#initialize variables
				sess.run(tf.global_variables_initializer(),feed_dict)
					
				#run the signal through the net
				result = sess.run(self.output, feed_dict)

			if recordGraph:
				writer.add_graph(sess.graph)
				writer.close()

		if self.isRecurrent:
			raise ValueError('untested')
			result = copy.deepcopy(presult[0])
			for step in range(1,n_steps):
				result = np.hstack((result,presult[step]))

		if onehot:
			Ys = np.argmax(result,axis=2)
			Yz = np.zeros(result.shape)
			for d1 in range(np.shape(Yz)[0]):
				for d2 in range(np.shape(Yz)[1]):
					ind = Ys[d1,d2]
					Yz[d1,d2,ind] = 1
			return Yz
		else:
			return result
		
	def Train(self,X: np.array,T: np.array,log_device_placement=False,recordGraph=False,recordPath='./Graphs/defaultGraphName',dynamic_preprocessing=True,use_regression=False):
		"""Trains this pynet given inputs and targets. If is a pynetgroup, each pynet in the group is trained individually and the group's performance function is used as the overall loss.
		Args:
			X (np.array): shape => [n_samples,n_features] or [n_members,n_samples,n_features]
			T (np.array): shape => [n_samples,n_classes] or [n_members,n_samples,n_classes]
			log_device_placement (bool): Print out  which devices your operations and tensors are assigned to.
			recordGraph (bool): Whether to record the network using tensorboard.
			recordPath (str):  Path to record network graph to.
			dynamic_preprocessing (bool): If false, self.preProcessor.settings must be set.
			use_regression (bool): Evaluate training using regression techniques. Default is classification.
		Returns:
			dict: training_result
		"""
			
		#determine if pynet is single net or group of nets
		isGroup,sample_dim,n_members = Pynet.Utils.DetermineGroup(self)
				
		#get shapes of members/samples/features/outputs
		input_shape = np.shape(X)
		target_shape = np.shape(T)

		if len(target_shape) == 1:
			T = np.expand_dims(T,1)
			target_shape = np.shape(T)
			
		self.n_features = input_shape[1]
		self.n_classes = target_shape[1]

		#cannot use classification default if only a single class
		if not use_regression and self.n_classes == 1:
			use_regression = True

		n_samples = input_shape[0]

		#set output layer n_nodes to correct n_classes
		self.layers[np.argmax(np.array(self.isOutput))].n_nodes = self.n_classes
			
		#split data into
		Xt,Xv,Xs,Tt,Tv,Ts = DatasetDivider.DNA.fcns[self.trainer.divideFcn](X,T,self.trainer.divideParams['trainRatio'],self.trainer.divideParams['valRatio'],self.trainer.divideParams['testRatio'])

		#vertical stack if workign on group
		if isGroup:

			#add member dim
			Xtd = np.expand_dims(Xt,axis=0)
			Xvd = np.expand_dims(Xv,axis=0)
			Xsd = np.expand_dims(Xs,axis=0)
			Ttd = np.expand_dims(Tt,axis=0)
			Tvd = np.expand_dims(Tv,axis=0)
			Tsd = np.expand_dims(Ts,axis=0)

			XXt = copy.deepcopy(Xtd)
			XXv = copy.deepcopy(Xvd)
			XXs = copy.deepcopy(Xsd)
			TTt = copy.deepcopy(Ttd)
			TTv = copy.deepcopy(Tvd)
			TTs = copy.deepcopy(Tsd)

			for member in range(0,n_members - 1):
				Xt,Xv,Xs,Tt,Tv,Ts = DatasetDivider.DNA.fcns[self.trainer.divideFcn](X,T,\
					self.trainer.divideParams['trainRatio'],\
					self.trainer.divideParams['valRatio'],\
					self.trainer.divideParams['testRatio'])

				Xtd = np.expand_dims(Xt,axis=0)
				Xvd = np.expand_dims(Xv,axis=0)
				Xsd = np.expand_dims(Xs,axis=0)
				Ttd = np.expand_dims(Tt,axis=0)
				Tvd = np.expand_dims(Tv,axis=0)
				Tsd = np.expand_dims(Ts,axis=0)

				XXt = np.vstack((XXt,Xtd))
				XXv = np.vstack((XXv,Xvd))
				XXs = np.vstack((XXs,Xsd))
				TTt = np.vstack((TTt,Ttd))
				TTv = np.vstack((TTv,Tvd))
				TTs = np.vstack((TTs,Tsd))

			training_dict = {
				'X_tf:0': XXt,
				'T_tf:0': TTt
				}
			validation_dict = {
				'X_tf:0': XXv,
				'T_tf:0': TTv
				}
			testing_dict = {
				'X_tf:0': XXs,
				'T_tf:0': TTs
				}
			Ts = TTs
			Xs = XXs
		else:
			raise ValueError('double check this')
			training_dict = {
				'X_tf:0': Xt,
				'T_tf:0': Tt
				}
			validation_dict = {
				'X_tf:0': Xv,
				'T_tf:0': Tv
				}
			testing_dict = {
				'X_tf:0': Xs,
				'T_tf:0': Ts
				}

		#2
		#initalize weights/bias via layer initialization functions
		for L in range(0,len(self.layers)):
			if isGroup:
				if self.isInput[L]:
					w_shape = [n_members,self.layers[L].n_nodes,self.n_features]
					b_shape = [n_members,self.layers[L].n_nodes]
					self.layers[L].n_inputs_to_layer = self.n_features
					w,b = Initialization.DNA.fcns[self.layers[L].initFcn](self.layers[L],w_shape,b_shape)
					self.layers[L].inputWeights = w

				if self.isOutput[L]:
					b_shape = [n_members,self.n_classes]
					for L_i in range(0,len(self.layers)):
						if self.layerConnect[L][L_i]:
							w_shape = [n_members,self.n_classes,self.layers[L_i].n_nodes]
							self.layers[L].n_inputs_to_layer = self.layers[L_i].n_nodes
							w,b = Initialization.DNA.fcns[self.layers[L].initFcn](self.layers[L],w_shape,b_shape)
							#lw = []
							#lw.append(w)
							self.layers[L].layerWeights[L_i] = w #TODO L_i not exist here due to dna2net
				else:
					b_shape = [n_members,self.layers[L].n_nodes]
					for L_i in range(0,len(self.layers)):
						if self.layerConnect[L][L_i]:
							w_shape = [n_members,self.layers[L].n_nodes,self.layers[L_i].n_nodes]
							self.layers[L].n_inputs_to_layer = self.layers[L_i].n_nodes
							w,b = Initialization.DNA.fcns[self.layers[L].initFcn](self.layers[L],w_shape,b_shape)
							#lw = []
							#lw.append(w)
							self.layers[L].layerWeights[L_i] = w
				#TODO: if no b here, then layer has no input/connection
				self.layers[L].bias = b #b should be correct no matter - unless connecting layer as input isnt correct size
			else:
				raise ValueError('double check this - not supporting non groups')
				if self.isInput[L]:
					w_shape = [self.layers[L].n_nodes,self.n_features]
					b_shape = [self.layers[L].n_nodes]
					self.layers[L].n_inputs_to_layer = self.n_features
					w,b = Initialization.DNA.fcns[self.layers[L].initFcn](self.layers[L],w_shape,b_shape)
					self.layers[L].inputWeights = w

				if self.isOutput[L]:
					b_shape = [self.n_classes]
					for L_i in range(0,len(self.layers)):
						if self.layerConnect[L][L_i]:
							w_shape = [self.n_classes,self.layers[L_i].n_nodes]
							self.layers[L].n_inputs_to_layer = self.layers[L_i].n_nodes
							w,b = Initialization.DNA.fcns[self.layers[L].initFcn](self.layers[L],w_shape,b_shape)
							lw = []
							lw.append(w)
							self.layers[L].layerWeights[L_i] = lw
				else:
					b_shape = [self.layers[L].n_nodes]
					for L_i in range(0,len(self.layers)):
						if self.layerConnect[L][L_i]:
							w_shape = [self.layers[L].n_nodes,self.layers[L_i].n_nodes]
							self.layers[L].n_inputs_to_layer = self.layers[L_i].n_nodes
							w,b = Initialization.DNA.fcns[self.layers[L].initFcn](self.layers[L],w_shape,b_shape)
							lw = []
							lw.append(w)
							self.layers[L].layerWeights[L_i] = lw
				self.layers[L].bias = b #b should be correct no matter - unless connecting layer as input isnt correct size

		#establish preprocessing settings from data
		if dynamic_preprocessing:
			self.preProcessor.settings.xmax = np.max(X,axis=0)
			self.preProcessor.settings.xmin = np.min(X,axis=0)
			self.preProcessor.settings.ymax = np.array([np.max(T)],np.float32)
			self.preProcessor.settings.ymin = np.array([np.min(T) - 1],np.float32) #-1 because iported is -1, but this result is 0 --> why? sometimes onehot data doesnt have 0?
			self.preProcessor.settings.xmean = np.mean(X,axis=0)
			self.preProcessor.settings.xstd = np.std(X,axis=0)

		self.ConfigureGraph()
			
		base_dict = {
			'xmax_tf:0': self.preProcessor.settings.xmax,
			'xmin_tf:0': self.preProcessor.settings.xmin,
			'ymax_tf:0': self.preProcessor.settings.ymax,
			'ymin_tf:0': self.preProcessor.settings.ymin,
			'xmean_tf:0': self.preProcessor.settings.xmean,
			'xstd_tf:0': self.preProcessor.settings.xstd,
			}

		validation_loss_last = 1e400
		validation_fails = 0

		gpuOptions = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=True,gpu_options=gpuOptions)
		with tf.Session(graph=self.graph,config=config) as sess:

			if recordGraph: writer = tf.summary.FileWriter(recordPath)

			#run session
			sess.run(tf.global_variables_initializer(),dict(base_dict,**training_dict))

			for epoch in range(1,self.trainer.epochs):

				#update weights/bias
				_,training_loss = sess.run([self.update_wb,self.error],feed_dict=dict(base_dict,**training_dict))

				#get validation error
				validation_loss = sess.run(self.error,feed_dict=dict(base_dict,**validation_dict))
					
				#has validation error decreased?
				if validation_loss < validation_loss_last:
					validation_loss_last = validation_loss

					#extract output of variables after updating over in pynet structure
					for L in range(0,len(self.layers)):
						if isGroup:
							self.layers[L].bias = np.squeeze(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'b' + str(L))[0]),axis=1)

							if self.isInput[L]:
								self.layers[L].inputWeights = np.transpose(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'iw' + str(L))[0]),axes=[0,2,1])

							for L_i in range(0,len(self.layers)):
								if self.layerConnect[L][L_i]:
									lw = np.transpose(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'Lw' + str(L) + 'i' + str(L_i))[0]),axes=[0,2,1])
									self.layers[L].layerWeights[L_i] = lw

						else:
							raise ValueError('double check this')
							self.layers[L].bias = np.transpose(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'b' + str(L))[0]))

							if self.isInput[L]:
								self.layers[L].inputWeights = np.transpose(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'iw' + str(L))[0]),axes=[1,0])

							for L_i in range(0,len(self.layers)):
								if self.layerConnect[L][L_i]:
									lw = []
									lw.append(np.transpose(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'Lw' + str(L) + 'i' + str(L_i))[0]),axes=[1,0]))
									self.layers[L].layerWeights[L_i] = lw

					validation_fails = 0
				else:
					#validation error increased, therefore a fail
					validation_fails = validation_fails + 1

				if self.trainer.showCommandLine:
					print('Epoch: %d, Validation Checks: %d' % (epoch,validation_fails))
					print('Training Loss: %f' % training_loss)
					print('Validation Loss: %f' % validation_loss)


				#validaton fail check
				if (validation_fails >= self.trainer.max_fail) | (validation_loss_last <= self.trainer.goal):
					break

			if recordGraph:
				writer.add_graph(sess.graph)
				writer.close()

			if self.trainer.showCommandLine:
				print('Epochs: %d' % epoch)
				print('Validation Checks: %d' % validation_fails)
				print('Validation Loss: %f' % validation_loss_last)

			#reconfigure for use
			self.ConfigureGraph()

			try:
				loss = validation_loss_last.tolist()
			except: #inf will be float not np.float32
				loss = validation_loss_last

			if not use_regression:
				scores = {
					'acc': [],
					'f1': [],
					'hamming': [],
					'prec': [],
					'recall': [],
					'auc_roc': [],
					'brier': [],
					'mcc': [],
					'jac': []
					}
			else:
				scores = {
					'mse': [],
					'mae': [],
					'msle': [],
					'r2': []
					}

			#simulate an output signal
			Y = self.Sim(X)

			for member in range(n_members):

				if not use_regression:
					#results evaluated using classificaiton scores

					y_pred = np.argmax(Y[member],axis=1).flatten()
					y_true = np.argmax(T,axis=1).flatten()

					y_prob = Y[member,:,:].flatten()
					y_class = T.flatten()

					scores['acc'].append(metrics.accuracy_score(y_true,y_pred))
					scores['f1'].append(metrics.f1_score(y_true,y_pred,average='macro'))
					scores['hamming'].append(metrics.hamming_loss(y_true,y_pred))
					scores['prec'].append(metrics.precision_score(y_true,y_pred,average='macro'))
					scores['recall'].append(metrics.recall_score(y_true,y_pred,average='macro'))
					scores['mcc'].append(metrics.matthews_corrcoef(y_true,y_pred))
					#scores['jac'].append(metrics.jaccard_similarity_score(y_true,y_pred))
					scores['jac'].append(metrics.jaccard_score(y_true,y_pred,average='macro'))
				
					if np.any(y_prob < 0) or np.any(y_prob > 1):
						scores['auc_roc'].append(0)
						scores['brier'].append(1)
					else:
						scores['auc_roc'].append(metrics.roc_auc_score(y_class,y_prob))
						scores['brier'].append(metrics.brier_score_loss(y_class,y_prob))
				else:
					#results evaulated using regression scores
					y_true = T.flatten()
					y_pred = Y[member].flatten()

					if any(np.isnan(y_pred)):
						scores['mse'].append(np.inf)
						scores['mae'].append(np.inf)
						scores['r2'].append(0)
						scores['msle'].append(np.inf)
					else:
						scores['mse'].append(metrics.mean_squared_error(y_true,y_pred))
						scores['mae'].append(metrics.mean_absolute_error(y_true,y_pred))
						scores['r2'].append(metrics.r2_score(y_true,y_pred))
						
						try:
							scores['msle'].append(metrics.mean_squared_log_error(y_true,y_pred))
						except:
							scores['msle'].append(np.inf)

			training_results = {
				'epochs': epoch,
				'validation_fails': validation_fails,
				'loss': loss,
				'scores': scores
				}

			return training_results

	def Evolve(X: np.array,T: np.array,useGPU: bool,population_capacity: int,fitness_method: str,\
		elite_percentage=0.1,survivor_percentage=0.2,mutation_chance=0.001,seeds=None,groupSize=8,time_limit_minutes=60,max_gen=60,fitness_target=0.9999,use_regression=False):
		"""Genetically evolve a Pynet neural network to an optimal structure for the input/target data.
		Args:
			X (np.array): shape => [n_samples,n_features] or X[n_members,n_samples,n_features]
			T (np.array): shape => [n_samples,n_classes] or X[n_members,n_samples,n_classes]
			useGPU (bool): True to configure and utilize the GPU for computation.
			population_capacity (int): Total number of individul dnas.
			fitness_method (str): 'acc' | 'f1' | 'hamming' | 'prec' | 'recall' | 'mcc' | 'jac' | 'auc_roc' | 'brier' OR 'mse' | 'mae' | 'msle' | 'r2' (for regression)
			elite_percentage (float): Percentage of population considered to be elite and will automatically be selected to survive.
			survivor_percentage (float): Percentage of population that will survive.
			mutation_chance (float): Percent change of an allele to mutate (switch) during crossover.
			seeds (list): List of integer dnas to seed this population.
			groupSize (int): Number of members in the pynet group.
			time_limit_minutes (int): Default of 60 minute time limit for evolving.
			max_gen (int): Maximum generations for the evolving population to achieve.
			fitness_target (float): Target to achieve with the specified fitness method.
			use_regression (bool): Evaluate training using regression techniques. Default is classification.
		Returns:
			Pynet: net
		Ref:
			https://www.ijcai.org/Proceedings/89-1/Papers/122.pdf

		Notes:
			Evolving completes when the time limit expires, the max generation is reached in the population, or the fitness target is achieved - whichever occurs first.
		"""

		population = Population(**{
			'elite_percentage': elite_percentage,
			'survivor_percentage': survivor_percentage,
			'mutation_chance': mutation_chance,
			},seeds=seeds)

		if population.seeds is not None:
			#add seed to census
			for seed in population.seeds:
				population.census.append(seed)
				population.generated.append(True)

		#acquire shape of inputs
		input_shape = np.shape(X)
		target_shape = np.shape(T)
		
		n_samples = input_shape[0]
		n_features = input_shape[1]
		n_classes = target_shape[1]

		#total possible bits for dna
		total_possible_bits = Pynet.DNA.total_bits()

		#finish bringing census to capacity with random population
		while len(population.census) < population_capacity:
			dna = Pynet.DNA.Rand(n_classes)
			if Pynet.DNA.IsValid(dna,False): #TODO: allow recurrent?  not until sim/train can do sequential
				population.census.append(dna)
				try:
					population.parents[str(dna)]
				except:
					population.parents[str(dna)] = [None]
				population.generated.append(True)

		def get_fitness(method: str,result: dict):
			"""Get the fitness based upon the method.
			Note:
				In some methods, lower score is better but this evolution searches for highest so must be flipped.
			"""
			if (method == 'brier') or (method == 'hamming'):
				#brier sore and hamming are better the closer they are to 0, not 1
				fit_score = 1 - np.mean(result['scores'][method])
			elif (method == 'mse') or (method == 'mae') or (method == 'msle') :
				#regression errors are better the smaller they are
				a = 5
				fit_score = 1 - (1 / np.mean(result['scores'][method]))
				a = 5
			else:
				fit_score = np.mean(result['scores'][method])

			return fit_score

		#assess initial generation fitness
		for dna in population.census:
			training_result = Pynet.DNA.Form(dna,groupSize,useGPU,n_features,n_classes).Train(X,T,use_regression=use_regression)
			population.fitness.append(get_fitness(fitness_method,training_result))

		#store best dna
		best_dna = population.census[np.argsort(population.fitness)[-1]]
		best_fitness = population.fitness[np.argsort(population.fitness)[-1]]

		#produce next generation through survival, selection, and breeding
		clock = 0
		gen = 0
		t0 = time.time()
		while (clock <= time_limit_minutes * 60) and (gen <= max_gen) and (best_fitness < fitness_target): #TODO: stall

			#get new generation
			#TODO: is pop cap too small, wont be any survivors (all be elite?)
			population = Evolution.Selection(population,total_possible_bits,survivor_percentage,elite_percentage,mutation_chance)

			#ensure new generation has valid dna
			for individual in range(0,len(population.census)):
				dna = population.census[individual]

				#only allow valid dna, replace with a valid generated dna if needed
				generated = False
				while not Pynet.DNA.IsValid(dna,False):
					generated = True
					dna = Pynet.DNA.Rand(n_classes)

				population.census[individual] = dna

				if generated: population.generated[individual] = generated

			#train each dna in the population
			for dna in population.census:
				training_result = (Pynet.DNA.Form(dna,groupSize,useGPU,n_features,n_classes).Train(X,T))
				population.fitness.append(get_fitness(fitness_method,training_result))

			#store best
			new_best_fitness = population.fitness[np.argsort(population.fitness)[-1]]
			if new_best_fitness > best_fitness:
				best_dna = population.census[np.argsort(population.fitness)[-1]]
				best_fitness = population.fitness[np.argsort(population.fitness)[-1]]
			
			clock = time.time() - t0
			gen = gen + 1

		return Pynet.DNA.Form(best_dna,groupSize,useGPU,n_features,n_classes)

	def MemberOut(self,out: int):
		"""Pull a member out of this Pynet.
		Args:
			out (int): Member by index
		Returns:
			Pynet: member
		"""
		#extract own dna
		my_dna = Pynet.DNA.Extract(self)

		#create single member
		member = Pynet.DNA.Form(my_dna,1,False,n_features=self.n_features,n_classes=self.n_classes)
		#TODO: extract and forming could change DNA?

		#extract the layer's node values
		for L in range(len(self.layers)):
			try:
				member.layers[L].inputWeights[0] = self.layers[L].inputWeights[out]
			except:
				member.layers[L].inputWeights = self.layers[L].inputWeights

			for L_i in range(len(self.layers)):
				try:
					member.layers[L].layerWeights[L_i][0] = self.layers[L].layerWeights[L_i][out]
				except:
					member.layers[L].layerWeights[L_i] = self.layers[L].layerWeights[L_i]

			member.layers[L].bias[0] = self.layers[L].bias[out]
			
		return member

	def AppendMember(self,member):
		"""Append a Pynet to this Pynetgroup as a new member.
		Args:
			member (Pynet): 
		"""
		#extract the layer's node values
		for L in range(len(self.layers)):

			try:
				self.layers[L].inputWeights.append(member.layers[L].inputWeights[0])
			except:
				self.layers[L].inputWeights.append(member.layers[L].inputWeights)

			for L_i in range(len(self.layers)):
				try:
					self.layers[L].layerWeights[L_i].append(member.layers[L].layerWeights[L_i][0])
				except:
					self.layers[L].layerWeights[L_i].append(member.layers[L].layerWeights[L_i])
					
			self.layers[L].bias.append(member.layers[L].bias[0])

		return self

	class Utils:

		def IsRecurrent(net):
			"""Tests if a Pynet is a recurrent neural network that requires sequential input data.
			Args:
				net (Pynet): Pynet to test.
			Returns:
				bool: True if the pynet is recurrent.
			"""
			for L in range(0,len(net.layers)): #for every layer

				#check for layer connections
				for L_i in range(0, len(net.layers)): #from every layer
					if net.layerConnect[L][L_i]: #if layer L has connection from L_i
						if (L == L_i) or (L_i > L):
							return True
			return False

		def DetermineGroup(net):
			"""Determines if this pynet is a single neural network or a group calculated in parallel.
			Args:
				net (Pynet): Pynet to test.
			Returns:
				bool: isGroup
				int: sample_dim
				int: n_members
			Notes:
				Uses the shape of the bias vector of the first layer.
			"""

			#get the shape of the bias in the first layer
			b_shape = np.shape(net.layers[0].bias)
			if len(b_shape) == 1: #1d bias vector is a single net
				isGroup = False
				sample_dim = 0
				n_members = 1
			elif len(b_shape) == 2: #2d bias vector is a group
				isGroup = True
				sample_dim = 1
				n_members = b_shape[0]

			return isGroup,sample_dim,n_members

		def Initialize_layer_output_tensors(net,X_tf: tf.Tensor,isGroup: bool,n_members: int,sample_dim: int):
			"""Initialize layer output tensors so that they may be used as a layer connection input before the layer outputs a signal (feedback).
			Args:
				net (Pynet):Pynet to initialize layer outputs in.
				X_tf (tf.Tensor): Input data as a Tensor.
				isGroup (bool): If pynet is a group.
				n_members (int): Number of members in the pynet group.
				sample_dim (int): The dim whos length is the number of samples in the input.
			Returns:
				tf.Tensor: layer_output_tensors
			Notes:
				layer_output_tensors [n_members, n_samples, n_nodes] or [n_samples, n_nodes]
			"""

			layer_output_tensors = {}
			for L in range(0,len(net.layers)):
				if net.layerConnect[L][L]:
					#w_shape = [groupSize,n_nodes[L_i],n_nodes[L]] from dna2net
					#TODO: tf.shape is to be depricated? says .shape is always calculated
					n_samples = tf.shape(X_tf)[sample_dim]#self.layers[L].n_nodes #TODO: problem being none if recurrent - should
				else:
					n_samples = tf.shape(X_tf)[sample_dim]



				if isGroup:
					#1 - make a tensor of zeros
					layer_output_tensors[L] = tf.zeros([n_members,n_samples,net.layers[L].n_nodes],tf.float32,name='Ai_tf' + str(L))
					#FAIL - state doesnt persist outside session?

					#2 - make variable
					#layer_output_tensors[L] =
					#tf.Variable(tf.zeros([n_members,n_samples,net.layers[L].n_nodes]),\
					#	validate_shape=False,dtype=tf.float32,name='A_tf'+str(L))
					#FAIL - loss_tf cannot produce gradient

					#zeros_dim = tf.stack([n_members,n_samples,net.layers[L].n_nodes])
					#layer_output_tensors[L] = tf.fill(zeros_dim,0.0,name='A_tf'+str(L))

					#zeros_tf =
					#tf.zeros([n_members,n_samples,net.layers[L].n_nodes],tf.float32)
					#layer_output_tensors[L] =
					#tf.placeholder_with_default(zeros_tf,[n_members,None,net.layers[L].n_nodes],name='A_tf'+str(L))

					#layer_output_tensors[L] = tf.placeholder(tf.float32,name='A_tf'+str(L))
				else:
					#TODO: unchecked
					raise ValueError('no group is untested')
					layer_output_tensors[L] = tf.zeros(\
						[n_samples,\
						net.layers[L].n_nodes],tf.float32,\
						#validate_shape=False,\
						name='Ai_tf' + str(L))
			return layer_output_tensors

		def Initialize_layer_dependancies(net,Xp_tf: tf.Tensor,b: dict,iw: dict,lw: dict,layer_output_tensors: tf.Tensor):
			"""Initialize the variables required for each layer.
			Args:
				net (Pynet):Pynet to initialize layers dependancies in.
				X_tf (tf.Tensor): Input data as a Tensor.
				b (dict): Bias
				iw (dict): Input weights
				lw (dict): Layer weights
				layer_output_tensors (tf.Tensor): Previously initialized layer utput tensors.
			Returns:
				dict: layer_input_dependancies
				dict: layer_output_dependancies
			"""
			#create layer dependancies to ensure proper calculation control/flow
			layer_input_dependancies = {}
			layer_output_dependancies = {}
			for L in range(0,len(net.layers)): #for every layer
				layer_input_dependancies[L] = []
				layer_output_dependancies[L] = []
				layer_output_dependancies[L].append(b[L]) #must have bias tensor value initialized
					
				if net.isInput[L]: #if layer is connected to input
					layer_input_dependancies[L].append(iw[L]) #must have input weight initialized
					layer_input_dependancies[L].append(Xp_tf) #must have preprocessed input computed

				#check for layer connections
				for L_i in range(0, len(net.layers)): #from every layer
					if net.layerConnect[L][L_i]: #if layer L has connection from L_i
						layer_input_dependancies[L].append(lw[L][L_i]) #must have layer weight initialized
						layer_input_dependancies[L].append(layer_output_tensors[L_i]) #must have layer output initialized
			return layer_input_dependancies, layer_output_dependancies

		def Initialize_layer_outputs_feed_dict(net,isGroup,n_members,n_samples):
			"""DEPRECATED?
			
			"""
			raise ValueError('unused as of now?')
			layer_outputs_feed_dict = {}
			for L in range(0,len(net.layers)):
				if isGroup:
					layer_outputs_feed_dict['A_tf' + str(L) + ':0'] = np.zeros([n_members,n_samples,net.layers[L].n_nodes],np.float32)
				else:
					raise ValueError('double check this')
					layer_outputs_feed_dict['A_tf' + str(L) + ':0'] = np.zeros([n_samples,net.layers[L].n_nodes],np.float32)


			return layer_outputs_feed_dict
			
		def Build_Layer_tfVars(net,isGroup: bool):
			"""Build dicts of tensorflow Variables for input/layer weights and biases to compile the variable list for training.
			Args:
				net (Pynet):Pynet to initialize layers in.
				isGroup (bool): If pynet is a group.
			Returns:
				dict: input_weights
				dict: layer_weights
				dict: bias
				list: var_list
			Ref:
				https://stackoverflow.com/questions/44578992/how-to-update-the-variable-list-for-which-the-optimizer-need-to-update-in-tensor
				https://github.com/tensorflow/tensorflow/issues/834
				https://stackoverflow.com/questions/34477889/holding-variables-constant-during-optimizer/34478044
			TODO:
				Complete for if isGroup == False
			"""

			iw = {}
			lw = {}
			b = {}
			var_list = []
			for L in range(len(net.layers)):
				if isGroup:
					b[L] = tf.Variable(np.expand_dims(net.layers[L].bias,axis=1),dtype=tf.float32,name="b" + str(L))
					var_list.append(b[L])

					inputWeights = net.layers[L].inputWeights
					if np.size(inputWeights) > 0: #if layer is not connected to input, size == 0
						iw[L] = tf.Variable(np.transpose(inputWeights,axes=[0,2,1]),dtype=tf.float32,name="iw" + str(L))
						var_list.append(iw[L])

					layerWeights = net.layers[L].layerWeights
					lw[L] = {}
					for L_i in range(0, len(net.layers)):
						if np.size(layerWeights[L_i]) > 0:
							lw[L][L_i] = tf.Variable(np.transpose(layerWeights[L_i],axes=[0,2,1]),dtype=tf.float32,name="Lw" + str(L) + 'i' + str(L_i))
							var_list.append(lw[L][L_i])

				else:
					raise ValueError('double check this')
					b[L] = tf.Variable(np.transpose(net.layers[L].bias),dtype=tf.float32,name="b" + str(L))
					var_list.append(b[L])

					inputWeights = net.layers[L].inputWeights
					if np.size(inputWeights) > 0: #if layer is not connected to input, size == 0
						iw[L] = tf.Variable(np.transpose(inputWeights,axes=[1,0]),dtype=tf.float32,name="iw" + str(L))
						var_list.append(iw[L])

					layerWeights = net.layers[L].layerWeights
					lw[L] = {}
					for L_i in range(0, len(net.layers)):
						if np.size(layerWeights[L_i]) > 0:
							lw[L][L_i] = tf.Variable(np.transpose(layerWeights[L_i],axes=[1,0]),dtype=tf.float32,name="Lw" + str(L) + 'i' + str(L_i))
							var_list.append(lw[L][L_i])

			return iw,lw,b,var_list

		def X2feed_dict(net,X: np.array,isGroup: bool,n_members: int,sample_dim: int,feed_dict: dict):
			"""Builds feed_dict for Pynet.Sim from the X input.
			Args:
				net (Pynet):Pynet to initialize layers in.
				X (np.array): Input dataset.
				isGroup (bool): If pynet is a group.
				n_members (int): Number of members in the pynet group.
				sample_dim (int): The dim whos length is the number of samples in the input.
				feed_dict (dict): Dictionary of all tensors to be fed through pynet.
			Returns:
				dict: feed_dict
				int: n_steps:
				dict: step_dict
			"""

			#build feed dict
			if net.isRecurrent: #this net is recurrent and requires sequential step reading
				useXX = 0
				#vertical stack if working on group
				if isGroup:
					if len(np.shape(X)) == 3: #X data already presented for member consumption
						useXX = 1
					else: #x data presented for non group, stack to make ready for group consumption
						X = np.expand_dims(X,axis=0)
						XX = copy.deepcopy(X)
						for member in range(0,n_members - 1):
							XX = np.vstack((XX,X))
						useXX = 2
				else:
					useXX = 0

				#build sequential step dict
				step_dict = {}
				if n_members == 1:
					n_steps = np.shape(X)[sample_dim - 1]
				else:
					n_steps = np.shape(X)[sample_dim]
				for step in range(0,n_steps):
					if useXX == 0:
						step_dict[step] = X[step,:]
					if useXX == 1:
						step_dict[step] = X[:,step:step + 1,:]
					if useXX == 2:
						step_dict[step] = XX[:,step:step + 1,:]
			else: #not recurrent
				n_steps = 0
				step_dict = {}

				#vertical stack if working on group
				if isGroup:
					if len(np.shape(X)) == 3: #X data already presented for member consumption
						feed_dict['X_tf:0'] = X
					else: #x data presented for non group, stack to make ready for group consumption
						X = np.expand_dims(X,axis=0)
						XX = copy.deepcopy(X)
						for member in range(0,n_members - 1):
							XX = np.vstack((XX,X))
						feed_dict['X_tf:0'] = XX
				else:
					feed_dict['X_tf:0'] = X

			return feed_dict,n_steps,step_dict

	class DNA:
		"""Sub class of Pynet for managing genetic instructions.
		"""

		def total_bits():
			"""Computes the total bits required for a pynet dna.
			Returns:
				int: total_bits
			"""

			#get total bits of layer/node counts
			n_nodes_bits = Layer.DNA.n_nodes_bits()
			n_layers_bits = Layer.DNA.n_layer_bits()
			
			#overall param bits
			pre_bits = PreProcess.DNA.pre_process_bits + Performance.DNA.perform_fcn_bits + Training.DNA.train_fcn_bits + \
				Training.DNA.goal_bits + Training.DNA.max_fail_bits + Training.DNA.epoch_bits + Training.DNA.learning_rate_bits + \
				Training.DNA.min_grad_bits + DatasetDivider.DNA.divideFcn_bits + DatasetDivider.DNA.divideParams_bits + \
				Layer.DNA.n_layer_bits()
				
			#bits per layer
			#1 + 1: for isinput, isoutput, layer_connection
			layer_bits = Layer.DNA.layer_bits()
			
			#summate all bits
			total_bits = pre_bits + (Layer.DNA.n_layers_max * layer_bits)

			return total_bits

		def Form(dna: int,groupSize: int,useGPU: bool,n_features=4,n_classes=3):
			"""Form a pynet from a dna sequence.
			Args:
				dna (int): Integer representation of an entire pynet.
				groupSize (int): Number of members in the pynet group.
				useGPU (bool): True to configure and utilize the GPU for computation.
				n_features (int): Number of features or attributes for input datasets.
				n_classes (int): Number of classifications or outcomes for output dataset.
			Returns:
				Pynet: net
			Notes:
				n_features/classes only needs to be specified to initialize wb but is actually changed during training
			"""

			total_possible_bits = Pynet.DNA.total_bits()

			genes = npe.Int2Bin(dna,total_possible_bits)[::-1]

			#read the genes to extract the parameters
			next_gene_starts_idx = 0
				
			#pre processing fcn
			processFcn,next_gene_starts_idx = PreProcess.DNA.Read_Gene(genes,next_gene_starts_idx)

			#performance fcn
			performFcn,next_gene_starts_idx = Performance.DNA.Read_Gene(genes,next_gene_starts_idx)
				
			#training fcn
			trainFcn,goal,max_fail,epochs,learning_rate,min_grad,next_gene_starts_idx = Training.DNA.Read_Gene(genes,next_gene_starts_idx)

			#dataset divide fcn and didve params
			divideFcn,divideParams,next_gene_starts_idx = DatasetDivider.DNA.Read_Gene(genes,next_gene_starts_idx)

			#layers and input/output/connections
			layers,isInput,isOutput,layerConnect,next_gene_starts_idx = Layer.DNA.Read_Gene(groupSize,n_features,genes,next_gene_starts_idx)
				
			#create pynet
			preProcessor = PreProcess({
				'settings': processSettings({ #set in traing so ignore here
					'xmax': np.zeros([groupSize],np.float32),
					'xmin': np.zeros([groupSize],np.float32),
					'ymax': np.zeros([groupSize],np.float32),
					'ymin': np.zeros([groupSize],np.float32),
					'xmean': np.zeros([groupSize],np.float32),
					'xstd': np.zeros([groupSize],np.float32),
					}),
				'processFcn': processFcn
				})
			trainer = Training({
				'performFcn': performFcn,
				'trainFcn': trainFcn,
				'divideFcn': divideFcn,
				'divideParams': divideParams,
				'showCommandLine': False,
				'show': False,
				'epochs': epochs,
				'goal': goal,
				'min_grad': min_grad,
				'max_fail': max_fail,
				'learning_rate': learning_rate
				})
			
			#initialize pynet
			nnet = Pynet({
				'dna': dna,
				'preProcessor': preProcessor,
				'trainer': trainer,
				'isInput': isInput,
				'isOutput': isOutput,
				'layerConnect': layerConnect,
				'layers': layers,
				'n_features': n_features,
				'n_classes': n_classes,
				'onGPU': useGPU
				})
		
			nnet.isRecurrent = Pynet.Utils.IsRecurrent(nnet)
			return nnet

		def Extract(net):
			"""Extract the dna of a pynet.
			Args:
				net (Pynet): Target of dna extraction.
			Returns:
				int: dna
			"""	

			#process is to add binary strings of all components then turn that binary string into an integer
			genes = ''

			#preprocess fcn
			genes = genes + PreProcess.DNA.Write_Gene(net)

			#performance fcn
			genes = genes + Performance.DNA.Write_Gene(net)

			#train fcn
			genes = genes + Training.DNA.Write_Gene(net)

			#divide fcn and divide params
			genes = genes + DatasetDivider.DNA.Write_Gene(net.trainer.divideFcn,net.trainer.divideParams)

			#layer count
			genes = genes + npe.Int2Bin(len(net.layers),Layer.DNA.n_layer_bits())

			#for each layer, generate genes
			for layer in range(0,len(net.layers)):
				genes = genes + Layer.DNA.Write_Gene(net,layer)
			for unused_layer in range(0,(Layer.DNA.n_layers_max-len(net.layers))):
				genes = genes + (Layer.DNA.layer_bits() * '0')

			return npe.Bin2Int(genes[::-1])

		def IsValid(dna: int,recurrentIsValid: bool):
			"""Checks if a dna is valid by attempting to form a pynet, then validating parameters are within ranges.
			Args:
				dna (int): Integer representation of an entire pynet.
				recurrentIsValid (bool): If using a recurrent pynet is OK.
			Returns:
				bool: isValidPynet
			"""

			try:
				formed_net = Pynet.DNA.Form(dna,2,False,4,3) #TODO: are these set vals irrelevant/inconsequential?
			except Exception as e:
				#raise ValueError('Failed to form pynet from dna')
				return False

			#TODO: only want divide rand for now, ind not ready to be implimented
			if formed_net.trainer.divideFcn != 'dividerand':
				return False

			#validate ratios
			tp = formed_net.trainer.divideParams['trainRatio']
			vp = formed_net.trainer.divideParams['valRatio']
			sp = formed_net.trainer.divideParams['testRatio']
			if (np.round((tp + vp + sp),3) != 1.0):# or (tp < vp) or (tp < sp): #divide params not full
				return False

			#fcns int is within list len
			for L in range(0,len(formed_net.layers)):

				#if layer has no input or layer connect input, then its useless and dna is
				#invalid
				if (formed_net.isInput[L] + np.sum(formed_net.layerConnect[L])) == 0:
					return False

			#atleast 1 isInput
			if np.sum(formed_net.isInput) == 0:
				return False

			#only 1 isOutput, and atleast an output
			if (np.sum(formed_net.isOutput) > 1) or (np.sum(formed_net.isOutput) == 0):
				return False

			#TODO: if is input and isoutput and not layer input connections,
			#n_features/n_classes must be equal?
			#or n_nodes ?
			
			#layer connections isoutput is reachable from an isinput
			pathConnects = False
			isRecurrent = False
			uselessLayer = []
			n_layers = len(formed_net.layers)
			isInput = formed_net.isInput
			isOutput = formed_net.isOutput
			layerConnect = formed_net.layerConnect

			for L in range(0,n_layers):
				a = 5
				if isInput[L]: #if this layer takes input
					if isOutput[L]:
						pathConnects = True #this layer takes input and is output, path connects
						uselessLayer.append(False) #this layer is not useless
					else: #this layer doesnt output
						layerContributes = 0
						for L_o in range(0,n_layers):
							if layerConnect[L_o][L]: #if another layer takes signal from this layer
								layerContributes = layerContributes + 1 #this layer contributes
								if isOutput[L_o]:
									pathConnects = True #this layer contributes to a layer that is an output, path connects
									uselessLayer.append(False) #this layer is not useless
						if layerContributes == 0:
							uselessLayer.append(True) #this layer takes an input, but does not output nor contribute to other layer
				else: #this layer doesnt take input
					connectionsToLayer = 0
					contributionsByLayer = 0
					for L_i in range(0,n_layers):
						if layerConnect[L][L_i]: #this layer takes signal from another layer
							connectionsToLayer = connectionsToLayer + 1
							for L_o in range(0,n_layers):
								if layerConnect[L_o][L]: #this layer connects signal to another layer
									contributionsByLayer = contributionsByLayer + 1

					if (connectionsToLayer == 0):
						#this layer isnt connected to input, and takes no connections
						uselessLayer.append(True)

					if (contributionsByLayer + isOutput[L]) == 0:
						#this layer isnt connected to input, contributes to no other layer nor output
						uselessLayer.append(True)

			if np.sum(uselessLayer) > 0:
				return False #there is a useless layer

			if Pynet.Utils.IsRecurrent(formed_net):
				return False #TODO: recurrent currently not supported
				#if not recurrentIsValid:
				#	#TODO: if recurrent is valid and this is recurrent, divide fcn must be
				#	#indices
				#	return False

			return pathConnects

		def Rand(n_classes: int):
			"""Randomly generate dna for a Pynet.
			Args:
				n_classes (int): Number of classes/distinct features of the pynet to generate.
			Returns:
				int: dna
			"""

			genes = ''

			#need bits due to max nodes and layers
			n_nodes_bits = Layer.DNA.n_nodes_bits()
			n_layers_bits = Layer.DNA.n_layer_bits()

		
			#1) generate overarching network params
			#preprocess fcn
			preproc_gene = int(np.random.randint(len(PreProcess.DNA.fcns),size=1))
			genes = genes + npe.Int2Bin(preproc_gene,PreProcess.DNA.pre_process_bits)

			#performance fcn
			perf_gene = int(np.random.randint(len(Performance.DNA.fcns),size=1))
			genes = genes + npe.Int2Bin(perf_gene,Performance.DNA.perform_fcn_bits)

			#train fcn
			train_gene = int(np.random.randint(len(Training.DNA.fcns),size=1))
			genes = genes + npe.Int2Bin(train_gene,Training.DNA.train_fcn_bits)

			#train loss goal
			if Performance.DNA.genes[perf_gene] == 'crossentropy':
				goal_gene = 0 #cross entropy will depend on groupsize so goal is inconsequential
			else:
				goal_gene = int(np.random.randint(0,high=31,size=1)) #inverse of error goal ie 25 = 75% goal
			genes = genes + npe.Int2Bin(goal_gene,Training.DNA.goal_bits)

			#max validation fails
			max_fail_gene = int(np.random.randint(4,high=Training.DNA.max_fail_bits,size=1))
			genes = genes + npe.Int2Bin(max_fail_gene,Training.DNA.max_fail_bits)

			#epoch limit
			epoch_gene = int(np.random.randint(1000,high=2**Training.DNA.epoch_bits,size=1))
			genes = genes + npe.Int2Bin(epoch_gene,Training.DNA.epoch_bits)

			#learning rate
			lr_gene = int(np.random.randint(1,high=101,size=1))
			genes = genes + npe.Int2Bin(lr_gene,Training.DNA.learning_rate_bits)

			#min gradient epsilon
			ep_gene = int(np.random.randint(1,high=2**Training.DNA.min_grad_bits,size=1))
			genes = genes + npe.Int2Bin(ep_gene,Training.DNA.min_grad_bits)

			#divide Fcn
			divideFcn_gene = int(np.random.randint(len(DatasetDivider.DNA.fcns),size=1))
			genes = genes + npe.Int2Bin(divideFcn_gene,DatasetDivider.DNA.divideFcn_bits)

			#divide Params
			tp = 0;	vp = 0;	sp = 0
			while ((tp + vp + sp) != 100) or (tp < vp) or (tp < sp):
				tp = int(np.random.randint(5,high=91,size=1)) #max training is 90%
				vp = int(np.random.randint(5,high=(100 - tp - 4),size=1)) #max val leaves 1% for test
				sp = 100 - tp - vp
			genes = genes + npe.Int2Bin(tp,int(DatasetDivider.DNA.divideParams_bits / 3))
			genes = genes + npe.Int2Bin(vp,int(DatasetDivider.DNA.divideParams_bits / 3))
			genes = genes + npe.Int2Bin(sp,int(DatasetDivider.DNA.divideParams_bits / 3))


			#2) generate layer count gene
			n_layers = int(np.random.randint(2,Layer.DNA.n_layers_max + 1,size=1))
			genes = genes + npe.Int2Bin(n_layers,n_layers_bits)
		
			#3) for each layer, generate genes
			for layer in range(0,n_layers):

				#isinput
				isInput_gene = npe.Int2Bin(int(np.random.randint(2,size=1)),1)

				#isoutput
				isOutput = int(np.random.randint(2,size=1))
				isOutput_gene = npe.Int2Bin(isOutput,1)

				#layerconnect
				#layer_connect_gene = ''.join(str(e) for e in np.random.randint(2,size=Pynet.n_layers_max).tolist())

				layer_connect_gene = '0' * (Layer.DNA.n_layers_max-n_layers)
				for L_o in range(0,n_layers):
					layer_connect_gene = layer_connect_gene + npe.Int2Bin(int(np.random.randint(2,size=1)),1)
				layer_connect_gene = layer_connect_gene[::-1]

				#n_nodes
				n_nodes = 0
				if bool(isOutput):
					n_nodes = n_classes
				else:
					n_nodes = int(np.random.randint(1,Layer.DNA.n_nodes_max + 1,size=1))
				n_nodes_gene = npe.Int2Bin(n_nodes,n_nodes_bits)

				#initialization fcn
				init_gene = int(np.random.randint(len(Initialization.DNA.fcns),size=1))
				init_gene = npe.Int2Bin(init_gene,Initialization.DNA.init_fcn_bits)

				#weight fcn
				weight_gene = int(np.random.randint(len(Layer.DNA.weight_fcns),size=1))
				weight_gene = npe.Int2Bin(weight_gene,Layer.DNA.weight_fcn_bits)

				#input fcn
				input_gene = int(np.random.randint(len(Layer.DNA.input_fcns),size=1))
				input_gene = npe.Int2Bin(input_gene,Layer.DNA.input_fcn_bits)

				#transfer fcn
				transfer_gene = int(np.random.randint(len(Transfer.DNA.fcns),size=1))
				transfer_gene = npe.Int2Bin(transfer_gene,Transfer.DNA.transfer_fcn_bits)

				#append layer genes
				genes = genes + isInput_gene + isOutput_gene + layer_connect_gene + n_nodes_gene + init_gene + weight_gene + input_gene + transfer_gene

			#add the remaining non used layers as empty binary
			unused_layers = Layer.DNA.n_layers_max - n_layers
			if unused_layers > 0:
				for unused_layer in range(0,unused_layers):
					#create all '0000's thats length is determiend by
					
					genes = genes + ('0' * Layer.DNA.layer_bits())

			return npe.Bin2Int(genes[::-1])

	class IO:
		"""Managing Input/Output of a Pynet.
		"""

		def Load(jsonStr: str,onGPU: bool):
			"""Load a json str of the network into a Pynet
			Args:
				jsonStr (str): String of json.
				onGPU (bool) True to use graphs on GPU.
			Returns:
				Pynet: net
			"""
			
			try:
				jsonStr = jsonStr.replace('Inf','1e400').replace('Infinity','1e400')

				net = {}
				net = json.loads(jsonStr) #[1:len(jsonStr)-1])
			except:
				net = jsonStr
			
			#extract the preprocessing settings
			try:
				processFcn = net['inputs'][0]['processFcns'][0] #assume only 1 preprocessing function
				settings = processSettings({
					'xmax': np.array(net['inputs'][0]['processSettings'][0]['xmax'],np.float32),
					'xmin': np.array(net['inputs'][0]['processSettings'][0]['xmin'],np.float32),
					'ymax': np.array([net['inputs'][0]['processSettings'][0]['ymax']],np.float32),
					'ymin': np.array([net['inputs'][0]['processSettings'][0]['ymin']],np.float32),
					'xmean': np.array([net['inputs'][0]['processSettings'][0]['xmean']],np.float32),
					'xstd': np.array([net['inputs'][0]['processSettings'][0]['xstd']],np.float32)
					#TODO: xmean, xstd, xcov
					})
			except:
					
				#processFcn = net['inputs']['processFcns'][0] #assume only 1 preprocessing function
				settings = processSettings({
				#	'xmax': np.array(net['inputs']['processSettings']['xmax'],np.float32),
				#	'xmin': np.array(net['inputs']['processSettings']['xmin'],np.float32),
				#	'ymax': np.array([net['inputs']['processSettings']['ymax']],np.float32),
				#	'ymin': np.array([net['inputs']['processSettings']['ymin']],np.float32),
				#	'xmean': np.array([net['inputs']['processSettings']['xmean']],np.float32),
				#	'xstd': np.array([net['inputs']['processSettings']['xstd']],np.float32)
				#	})
				#raise ValueError('how get here? - failed to properly load preprocess settings?')
					'xmax': np.array(net['inputs'][0]['processSettings'][0]['xmax'],np.float32),
					'xmin': np.array(net['inputs'][0]['processSettings'][0]['xmin'],np.float32),
					'ymax': np.array([net['inputs'][0]['processSettings'][0]['ymax']],np.float32),
					'ymin': np.array([net['inputs'][0]['processSettings'][0]['ymin']],np.float32),
					'xmean': np.array([0],np.float32),
					'xstd': np.array([1],np.float32)
					})
			
			preProcessor = PreProcess({
				'settings': settings,
				'processFcn': processFcn
				})
			trainer = Training({
				'performFcn': net['performFcn'],
				'trainFcn': net['trainFcn'],
				'divideFcn': net['divideFcn'],
				'divideParams': net['divideParam'],
				'showCommandLine': net['trainParam']['showCommandLine'],
				'show': net['trainParam']['show'],
				'epochs': net['trainParam']['epochs'],
				#'time': net['trainParam']['time'],
				'goal': net['trainParam']['goal'],
				'min_grad': net['trainParam']['min_grad'],
				'max_fail': net['trainParam']['max_fail'],
				#'sigma': net['trainParam']['sigma'],
				#'lambda': net['trainParam']['lambda']
				'learning_rate': 0.01 #TODO: net['biases'][L]['learnParam']['lr']
				})

			#extract params for each layer
			n_layers = net['numLayers']
			isInput = net['inputConnect']
			isOutput = net['outputConnect']
			layerConnect = net['layerConnect']
			
			#build each layer
			layers = {}
			for L in range(0,n_layers):
				#transfer fcn, input fcn, initilizing fcn, number of nodes, bias, input
				#weights, layer weights
				transferFcn = net['layers'][L]['transferFcn']
				inputFcn = net['layers'][L]['netInputFcn']
				initFcn = net['layers'][L]['initFcn']
				bias = np.array(net['b'][L],np.float32)
				if isInput[L]:
					weightFcn = net['inputWeights'][L]['weightFcn']
					#n_nodes = np.shape(np.array(net['IW'][L],np.float32))[0]
				else:
					weightFcn = net['layerWeights'][L]['weightFcn']
					#n_nodes = np.shape(np.array(net['LW'][L],np.float32))[0]
				#TODO weight fcn diff for input/layer

				inputWeights = np.array(net['IW'][L],np.float32)
				layerWeights = {}
				for L_i in range(0,n_layers): #TODO: fails if improperly saved?
					#layerWeights[L_i] = np.array(net['LW'][(L_i * n_layers) + L],np.float32), #list, not array
					lw = np.array(net['LW'][(L_i * n_layers) + L],np.float32)
					layerWeights[L_i] = lw #list, not array

				layers[L] = Layer(**{
					'transferFcn': transferFcn,
					'inputWeights': inputWeights,
					'layerWeights': layerWeights,
					'weightFcn': weightFcn,
					'bias': bias,
					'inputFcn': inputFcn,
					'initFcn': initFcn,
					'n_nodes': net['layers'][L]['size'] #np.shape(np.array(net['IW'][L],np.float32))[0]
					})

			#initialize pynet
			nnet = Pynet({
				'preProcessor': preProcessor,
				'trainer': trainer,
				'isInput': isInput,
				'isOutput': isOutput,
				'layerConnect': layerConnect,
				'layers': layers,
				#'adaptFcn': net['adaptFcn'],
				'n_features': net['inputs'][0]['size'],
				'n_classes': net['outputs'][n_layers - 1]['size'],
				'onGPU': onGPU
				})

			try:
				nnet.dna = net['dna']
			except:
				nnet.dna = Pynet.DNA.Extract(nnet)


			return nnet

	class Models:
		"""Premade Pynet Models by their DNA.
		"""

		patternnet_dna = 75325223835347814544659183344447907190322609631015425015808
		fitnet_dna = 25108409952254368433972867958786575861503766075303148945408

		def Seedify(groupSize: int,useGPU: bool):
			"""Makes all available models as seeds for an evolutionary population.
			Args:
				groupSize (int): Number of members in the pynet group.
				useGPU (bool): True to configure and utilize the GPU for computation.
			Returns:
				list: seeds
			"""

			patternnet = Pynet.Models.PatternnetGroup(groupSize,useGPU)
			fitnet = Pynet.Models.FitnetGroup(groupSize,useGPU)
			cascadenet = Pynet.Models.CascadenetGroup(groupSize,useGPU)

			seeds = []
			seeds.append(patternnet)
			seeds.append(fitnet)
			seeds.append(cascadenet)

			return seeds

		def PatternnetGroup(groupSize: int,useGPU: bool):
			"""Creates a feedforward network structure with rank3 characteristics to simulate a group of Matlab's Patternnet.
			Args:
				groupSize (int): Number of members in the pynet group.
				useGPU (bool): True to configure and utilize the GPU for computation.
			Returns:
				Pynet: patternnet
			Ref:
				https://www.mathworks.com/help/nnet/ref/patternnet.html
			"""
			return Pynet.DNA.Form(Pynet.Models.patternnet_dna,groupSize,useGPU)

		def FitnetGroup(groupSize: int,useGPU: bool):
			"""Creates a feedforward network structure with rank3 characteristics to simulate a group of Matlab's Fitnet.
			Args:
				groupSize (int): Number of members in the pynet group.
				useGPU (bool): True to configure and utilize the GPU for computation.
			Returns:
				Pynet: fitnet
			Ref:
				https://www.mathworks.com/help/deeplearning/ref/fitnet.html
			"""
			return Pynet.DNA.Form(Pynet.Models.fitnet_dna,groupSize,useGPU)

		def CascadenetGroup(groupSize: int,useGPU: bool):
			"""Creates a feedforward network structure with rank3 characteristics to simulate a group of Matlab's Fitnet.
			Args:
				groupSize (int): Number of members in the pynet group.
				useGPU (bool): True to configure and utilize the GPU for computation.
			Returns:
				Pynet: cascadenet
			Ref:
				https://www.mathworks.com/help/deeplearning/ref/cascadeforwardnet.html
			"""

			fitnet = Pynet.Models.FitnetGroup(groupSize,useGPU)
			fitnet.isInput[-1] = True

			return fitnet

class Training:
	"""Configuration of parameters which determine Pynet training procedure.
	Args:
		performFcn (str): see Performance.DNA.fcns.keys()
		trainFcn (str): see Training.DNA.fcns.keys()
		divideFcn (str): see DatasetDivider.DNA.fcns.keys()
		divideParams (dict): ['trainRatio'], ['valRatio'], ['testRatio']
		epochs (int): Max iterations for the training function.
		goal (float): Target value for performance function.
		min_grad (float): Minimum step to take with gradients.
		max_fail (int): Maximum amount of times to fail a validation check in a row before ending training.
		learning_rate (float): How much weights are adjusted after with respect o performance loss.
		showCommandLine (bool): Show training prinouts in command line.
	TODO:
		Capture more params?
		https://www.mathworks.com/help/nnet/ug/choose-a-multilayer-neural-network-training-function.html
		https://www.mathworks.com/help/nnet/ug/train-and-apply-multilayer-neural-networks.html
		https://www.mathworks.com/help/nnet/ug/train-neural-networks-with-error-weights-1.html
		https://www.mathworks.com/help/nnet/ug/neural-network-training-concepts.html#bss326e-2
		Add more training functions -> https://www.mathworks.com/help/deeplearning/ug/train-and-apply-multilayer-neural-networks.html
	"""
	def __init__(self, kwargs):

		self.performFcn = kwargs['performFcn']
		self.trainFcn = kwargs['trainFcn']
		self.divideFcn = kwargs['divideFcn']
		self.divideParams = kwargs['divideParams']
		self.epochs = kwargs['epochs']
		#self.time = kwargs['time']
		self.goal = kwargs['goal']
		self.min_grad = kwargs['min_grad'] #TODO: use this as episolin in training?
		self.max_fail = kwargs['max_fail']
		#self.sigma = kwargs['sigma']
		#self.lambd = kwargs['lambda']
		self.learning_rate = kwargs['learning_rate']

		#TODO: remove show?
		self.showCommandLine = kwargs['showCommandLine']
		#self.show = kwargs['show']
			
		return super().__init__()
	
	def Adam(min_grad: float,lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Adam optimization.
		Args:
			min_grad (): Minimum gradient allowed; epsilon to prevent NAN loss.
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
		"""
		return tf.train.AdamOptimizer(learning_rate=lr,epsilon=min_grad).minimize(loss_tf,var_list=var_list)

	def GradientDescent(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Gradient descent optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
			https://www.mathworks.com/help/nnet/ref/traingd.html
		"""
		return tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	def AdaptiveGradientDescent(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Adaptive gradient descent optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer
		"""
		return tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	def AdaptiveDeltaGradientDescent(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Adaptive Delta gradient descent optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer
		"""
		return tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	def FollowTheRegularizedLeader(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Follow the regularized leader optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer
		"""
		return tf.train.FtrlOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	def ProximalAdaptiveGradientDescent(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Proximal adaptive gradient descent optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/ProximalAdagradOptimizer
		"""
		return tf.train.ProximalAdagradOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	def ProximalGradientDescent(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Proximal gradient descent optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/ProximalGradientDescentOptimizer
		"""
		return tf.train.ProximalGradientDescentOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	def RootMeanSquarePropigation(lr: float,loss_tf: tf.Tensor,var_list: list):
		"""Root mean square optimization.
		Args:
			lr (float): Learning rate.
			loss_tf (tf.Tensor): The performance loss result.
			var_list (list): Optional list Variable objects to update to minimize loss.
		Returns:
			tf.Operation: update_wb
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
		"""
		return tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss_tf,var_list=var_list)

	class DNA:

		fcns = {
			'trainadam': lambda min_grad,lr,loss,var_list: Training.Adam(min_grad,lr,loss,var_list),
			'traingd': lambda min_grad,lr,loss,var_list: Training.GradientDescent(lr,loss,var_list),
			'trainagd': lambda min_grad,lr,loss,var_list: Training.AdaptiveGradientDescent(lr,loss,var_list),
			'trainadeltagd': lambda min_grad,lr,loss,var_list: Training.AdaptiveDeltaGradientDescent(lr,loss,var_list),
			'trainftrl': lambda min_grad,lr,loss,var_list: Training.FollowTheRegularizedLeader(lr,loss,var_list),
			'trainpagd': lambda min_grad,lr,loss,var_list: Training.ProximalAdaptiveGradientDescent(lr,loss,var_list),
			'trainpgd': lambda min_grad,lr,loss,var_list: Training.ProximalGradientDescent(lr,loss,var_list),
			'trainrms': lambda min_grad,lr,loss,var_list: Training.RootMeanSquarePropigation(lr,loss,var_list),
			#'trainscg': lambda loss,var_list: Training.Adam(min_grad,lr,loss,var_list) #TODO: make scaled conjugate gradient
			}

		genes = {
			0: 'trainadam',
			1: 'traingd',
			2: 'trainagd',
			3: 'trainadeltagd',
			4: 'trainftrl',
			5: 'trainpagd',
			6: 'trainpgd',
			7: 'trainrms',
			}

		train_fcn_bits = 8
		goal_bits = 8
		max_fail_bits = 8
		epoch_bits = 12
		learning_rate_bits = 10
		min_grad_bits = 10

		def Write_Gene(net: Pynet):
			"""Reads the pynet and returns the gene binary string for the training function.
			Args:
				net (Pynet): 
			Returns:
				str: gene
			Notes:
				goal is multiplied by 100 to preserve decimals
				learning_rate is multiplied by 1000 to preserve decimals
				min_grad is multiplied by 1000000 to preserve decimals
				why is rounding necessary? int doesnt round -> int(.29*100) == 28
			"""

			train_gene = list(Training.DNA.genes.keys())[list(Training.DNA.genes.values()).index(net.trainer.trainFcn)]
			gene =  npe.Int2Bin(train_gene,Training.DNA.train_fcn_bits) + \
				npe.Int2Bin(int(np.round(net.trainer.goal * 100)),Training.DNA.goal_bits) + \
				npe.Int2Bin(net.trainer.max_fail,Training.DNA.max_fail_bits) + \
				npe.Int2Bin(net.trainer.epochs,Training.DNA.epoch_bits) + \
				npe.Int2Bin(int(np.round(net.trainer.learning_rate*1000)),Training.DNA.learning_rate_bits) + \
				npe.Int2Bin(int(np.round(net.trainer.min_grad*1000000)),Training.DNA.min_grad_bits)

			return gene

		def Read_Gene(genes: str,next_gene_starts_idx: int):
			"""Reads the net gene binary string.
			Args:
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the datasetdivider allele begins in the genes.
			Returns
				str: trainFcn
				int: max_fail
				int: epochs
				float: learning_rate
				float: min_grad
				int: next_gene_starts_idx
			"""

			next_gene_ends_idx = next_gene_starts_idx + Training.DNA.train_fcn_bits
			trainFcn = Training.DNA.genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + Training.DNA.goal_bits
			goal = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx]) / 100
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + Training.DNA.max_fail_bits
			max_fail = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + Training.DNA.epoch_bits
			epochs = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + Training.DNA.learning_rate_bits
			learning_rate = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx]) / 1000
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + Training.DNA.min_grad_bits
			min_grad = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx]) / 1000000
			next_gene_starts_idx = next_gene_ends_idx

			return trainFcn,goal,max_fail,epochs,learning_rate,min_grad,next_gene_starts_idx

class Layer:
	"""Neuron layer for a Pynet.
	Args:
		transferFcn (str): see Transfer.DNA.keys()
		inputWeights (np.array): Weights for transferring input signals.
		layerWeights (np.array): Weights for transferring layer signals.
		weightFcn (str): 'dotprod' |
		bias (np.array): Bias for transfering layer output signal.
		inputFcn (str): 'netsum' |
		initFcn (str): 'initrand' | 'initnw'
		n_nodes (int): Number of neurons/nodes in the layer.
	"""
	def __init__(self,**kwargs):

		#signal transfer function out of layer
		self.transferFcn = kwargs['transferFcn']

		#layer weights and activation function
		self.inputWeights = kwargs['inputWeights']
		self.layerWeights = kwargs['layerWeights']
		self.weightFcn = kwargs['weightFcn']
			
		#layer bias and activation function
		self.bias = kwargs['bias']
		self.inputFcn = kwargs['inputFcn']

		#layer initializing weights/bias function
		self.initFcn = kwargs['initFcn']

		#count of nodes in layer
		self.n_nodes = kwargs['n_nodes']

		#number of inputs to layer
		self.n_inputs_to_layer = None

		#last state of the layer output (for recurrent models)
		self.lastState = None

		return super().__init__()

	def Dotprod(signal_tf: tf.Tensor,w_tf: tf.Tensor):
		"""Activation computation function which uses the dot product of the signal and layer weights
		Args:
			signal_tf (tf.Tensor): Xp_tf or A_tf -> Either preprocessed signal into Pynet or output signal from previous Layer.
			w_tf (tf.Tensor): Tensor of layer weight values.
		Returns:
			tf.Tensor: Z_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/dotprod.html
		"""
		with tf.name_scope('Dotprod'):
			#return tf.einsum('aij,ajk->aik',signal_tf,w_tf)
			return tf.matmul(signal_tf,w_tf)

	def Netsum(Z_tf_List: list,b_tf: tf.Tensor):
		"""Input computation function which summates the input signals and the layer biases
		Args:
			Z_tf_List (list): List of tf.Tensors produced by the weight function.
			b_tf (tf.Tensor): Tensor of layer bias values.
		Returns:
			tf.Tensor: N_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/netsum.html
		"""
		with tf.name_scope('Netsum'):
			n_tf = tf.add_n(Z_tf_List)
			#n_tf = tf.Print(n_tf, [n_tf], message="--- n_tf:\n")
			return tf.add(n_tf, b_tf)

	class DNA:

		weight_fcns = {
			'dotprod': lambda x,w: Layer.Dotprod(x,w)
			}
		weight_genes = {
			0: 'dotprod'
			}

		#help nnnetinput
		input_fcns = {
			'netsum': lambda z,b: Layer.Netsum(z,b)
			}
		input_genes = {
			0: 'netsum'
			}
		
		#maximum layers
		n_layers_max = 7 #must be a power of 2 sans 1

		#maximum nodes in a layer
		n_nodes_max = 31 #must be a power of 2 sans 1

		weight_fcn_bits = 8
		input_fcn_bits = 8

		def n_layer_bits():
			"""Number of bits to hold n_layers info
			Returns:
				int: n_layer_bits
			"""
			return int(np.floor(np.log2(Layer.DNA.n_layers_max) + 1))

		def n_nodes_bits():
			"""Number of bits to hold n_nodes info
			Returns:
				int: n_node_bits
			"""
			return int(np.floor(np.log2(Layer.DNA.n_nodes_max) + 1))

		def layer_bits():
			"""Number of bits to hold total layers info
			Returns:
				int: layer_bits
			"""

			#+1 for isinput
			#+1 for isoutput
			#Layer.n_layers_max for layerconnect

			return 1 + 1 + Layer.DNA.n_layers_max + Layer.DNA.n_nodes_bits() + Initialization.DNA.init_fcn_bits + Layer.DNA.weight_fcn_bits + Layer.DNA.input_fcn_bits + Transfer.DNA.transfer_fcn_bits
		
		def Write_Gene(net: Pynet,layer_idx: int):
			"""Writes the pynet's Layer as a binary string.
			Args:
				net (Pynet): Pynet to write the layer gene of.
				layer_idx (int): Index in the Pynet of the layer to write the genes of.
			Returns:
				str: gene
			"""

			#isinput
			isInput_gene = npe.Int2Bin(int(net.isInput[layer_idx]),1)

			#isoutput
			isOutput_gene = npe.Int2Bin(int(net.isOutput[layer_idx]),1)

			#layerconnect
			layer_connect_gene = '0' * (Layer.DNA.n_layers_max-len(net.layers))
			for L_o in range(0,len(net.layers)): 
				layer_connect_gene = layer_connect_gene + npe.Int2Bin(int(net.layerConnect[layer_idx][L_o]),1)
			layer_connect_gene = layer_connect_gene[::-1]

			#n_nodes
			n_nodes_gene = npe.Int2Bin(net.layers[layer_idx].n_nodes,Layer.DNA.n_nodes_bits())

			#initialization fcn
			init_gene = Initialization.DNA.Write_Gene(net,layer_idx)

			#weight fcn
			weight_gene = npe.Int2Bin(list(Layer.DNA.weight_genes.keys())[list(Layer.DNA.weight_genes.values()).index(net.layers[layer_idx].weightFcn)],Layer.DNA.weight_fcn_bits)

			#input fcn
			input_gene = npe.Int2Bin(list(Layer.DNA.input_genes.keys())[list(Layer.DNA.input_genes.values()).index(net.layers[layer_idx].inputFcn)],Layer.DNA.input_fcn_bits)

			#transfer fcn
			transfer_gene = Transfer.DNA.Write_Gene(net,layer_idx)

			#append layer genes
			return isInput_gene + isOutput_gene + layer_connect_gene + n_nodes_gene + init_gene + weight_gene + input_gene + transfer_gene

		def Read_Gene(groupSize: int,n_features: int,genes: str,next_gene_starts_idx: int):
			"""Reads the net gene binary string.
			Args:
				groupSize (int): Size of group in Pynet.
				n_features (int): Number of features in the Input/target dataset.
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the layer allele begins in the genes.
			Returns:
				list: layers
				list: isInput
				list: isOutput
				list: layerConnect
				int: next_gene_starts_idx
			Notes:
				The bias' and weights, when read, are set to zero.
			"""

			next_gene_ends_idx = next_gene_starts_idx + Layer.DNA.n_layer_bits()
			n_layers = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
			next_gene_starts_idx = next_gene_ends_idx

			if n_layers < 2: raise ValueError('n_layers < 2, invalid Pynet') #TODO: minimum 2 layers

			isInput = []
			isOutput = []
			layerConnect = []
			n_nodes = {}
			initFcn = {}
			weightFcn = {}
			inputFcn = {}
			transferFcn = {}

			#loop over remaining bits to build layers
			for L in range(0,n_layers):
				
				#isinput
				next_gene_ends_idx = next_gene_starts_idx + 1 #1 bit
				isInput.append(bool(npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])))
				next_gene_starts_idx = next_gene_ends_idx

				#isoutput
				next_gene_ends_idx = next_gene_starts_idx + 1 #1 bit
				isOutput.append(bool(npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])))
				next_gene_starts_idx = next_gene_ends_idx

				#layerconnect
				next_gene_ends_idx = next_gene_starts_idx + Layer.DNA.n_layers_max
				layer_connect_gene = genes[next_gene_starts_idx:next_gene_ends_idx]
				layer_connect_gene = layer_connect_gene[::-1] #why was this off?
				layer_connect_gene = layer_connect_gene[-n_layers:]
				next_gene_starts_idx = next_gene_ends_idx
				layer_input = []
				for L_i in range(0,n_layers):
					layer_input.append(bool(npe.Bin2Int(layer_connect_gene[L_i])))
				layerConnect.append(layer_input)

				next_gene_ends_idx = next_gene_starts_idx + Layer.DNA.n_nodes_bits()
				n_nodes[L] = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
				next_gene_starts_idx = next_gene_ends_idx

				#layer initialization fcn
				initFcn[L],next_gene_starts_idx = Initialization.DNA.Read_Gene(genes,next_gene_starts_idx)

				#layer weight fcn
				next_gene_ends_idx = next_gene_starts_idx + Layer.DNA.weight_fcn_bits
				weightFcn[L] = Layer.DNA.weight_genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
				next_gene_starts_idx = next_gene_ends_idx

				#layer input fcn
				next_gene_ends_idx = next_gene_starts_idx + Layer.DNA.input_fcn_bits
				inputFcn[L] = Layer.DNA.input_genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
				next_gene_starts_idx = next_gene_ends_idx

				transferFcn[L],next_gene_starts_idx = Transfer.DNA.Read_Gene(genes,next_gene_starts_idx)

			#build the layers from the genes
			layers = {}
			inputWeights = {}
			layerWeights = {}
			for L in range(0,n_layers):
				b_shape = [groupSize,n_nodes[L]]

				if isInput[L]:
					#inputWeights[L] = np.zeros([groupSize,n_features,n_nodes[L]],np.float32) #orig, should be flipped pre transpose?
					inputWeights[L] = np.zeros([groupSize,n_nodes[L],n_features],np.float32) #pre transpose?
				else:
					inputWeights[L] = []
				
				layerWeights[L] = []
				for L_i in range(0,n_layers):
					if layerConnect[L][L_i]:
						#w_shape = [groupSize,n_nodes[L_i],n_nodes[L]] #orig, should be flipped #as pre transpose?
						w_shape = [groupSize,n_nodes[L],n_nodes[L_i]] #pre transpose
						layerWeights[L].append(np.zeros(w_shape,np.float32))
					else:
						layerWeights[L].append([])

				layers[L] = Layer(**{
					'transferFcn': transferFcn[L],
					'inputWeights': inputWeights[L],
					'layerWeights': layerWeights[L],
					'weightFcn': weightFcn[L],
					'bias': np.zeros(b_shape,np.float32), #TODO: size mattes for determining if isgroup in configure
					'inputFcn': inputFcn[L],
					'initFcn': initFcn[L],
					'n_nodes': n_nodes[L]
					})

			return layers,isInput,isOutput,layerConnect,next_gene_starts_idx

class Initialization:
	"""Manages the initialization of the bias' and weights of a layer.
	"""

	def Random(w_shape: list,b_shape: list):
		"""Initializes weights and biases according to a random uniform distribution.
		Args:
			w_shape (list): Shape of the layer weights.
			b_shape (list): Shape of the layer bias'.
		Returns:
			np.array: weights
			np.array: bias
		"""
		input_range = [-1,1]
		weights =  np.random.uniform(low=input_range[0],high=input_range[1],size=w_shape)
		bias = np.random.uniform(low=input_range[0],high=input_range[1],size=b_shape)
		return weights,bias

	def NguyenWidrow(layer: Layer,w_shape: list,b_shape: list):
		"""Initializes weights and biases according to the Nguyen-Widrow algorythm.
		Args:
			layer (Layer): Layer to initialize w/b of.
			w_shape (list): Shape of weights
			b_shape (list): Shape of bias
		Returns:
			np.array: m_w
			np.array: m_b
		Ref:
			https://web.stanford.edu/class/ee373b/nninitialization.pdf
			https://pythonhosted.org/neurolab/_modules/neurolab/init.html
		"""

		input_range = Transfer.DNA.input_range[layer.transferFcn]
		output_range = Transfer.DNA.output_range[layer.transferFcn]

		n_nodes = layer.n_nodes
		n_inputs_to_layer = layer.n_inputs_to_layer

		#1) check if input_range isfinite, layer uses dotprod and netsum, otherwise do rand
		if np.any(np.isinf(input_range)) or \
			(not layer.inputFcn == 'netsum') or \
			(not layer.weightFcn == 'dotprod'):
			return Initialization.Random(w_shape,b_shape)
		
		#2) force spread across hardlim(s)
		if input_range[0] == input_range[1]:
			input_range[0] = input_range[0] - 1
			input_range[1] = input_range[1] + 1

		#3) handle special caseof no inputs
		#set w,b to zeros?

		#4) repeat to reshape output range
		output_range = np.tile(output_range,(n_inputs_to_layer,1))

		#5) calculate nw method
		def calcnw(input_range,output_range,n_inputs_to_layer,n_nodes):
			ci = n_inputs_to_layer
			cn = n_nodes
			w_fix = 0.7 * cn ** (1. / ci)
			w_rand = np.random.rand(cn, ci) * 2 - 1
			# Normalize
			if ci == 1:
				w_rand = w_rand / np.abs(w_rand)
			else:
				w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1)) * w_rand

			w = w_fix * w_rand
			b = np.array([0]) if cn == 1 else w_fix * np.linspace(-1, 1, cn) * np.sign(w[:, 0])

			# Scaleble to inp_active
			amin, amax = input_range[0],input_range[1]

			x = 0.5 * (amax - amin)
			y = 0.5 * (amax + amin)
			w = x * w
			b = x * b + y

			# Scaleble to inp_minmax
			minmax = output_range

			x = 2. / (minmax[:, 1] - minmax[:, 0])
			y = 1. - minmax[:, 1] * x
			w = w * x
			b = np.dot(w, y) + b

			return w,b

		#initialize return weights/bias
		m_w = np.zeros(w_shape)
		m_b = np.zeros(b_shape)
		
		#if isgroup then first dim is members
		if len(w_shape) > 2:
			#for each member, calc nw w/b
			for member in range(0,w_shape[0]):
				m_w[member], m_b[member] = calcnw(input_range,output_range,n_inputs_to_layer,n_nodes)
		else:
			#just calc nw for w/b
			m_w, m_b = calcnw(input_range,output_range,n_inputs_to_layer,n_nodes)

		return m_w,m_b

	class DNA:

		fcns = {
			'initrand': lambda layer,w_shape,b_shape: Initialization.Random(w_shape,b_shape),
			'initnw': lambda layer,w_shape,b_shape: Initialization.NguyenWidrow(layer,w_shape,b_shape)
			}
		genes = { 
			0: 'initrand', 
			1: 'initnw'
			}

		init_fcn_bits = 8

		def Write_Gene(net: Pynet,layer_idx: int):
			"""Writes the pynet layer's Initialization Function as a binary string.
			Args:
				net (Pynet): Pynet to write the layer initialization gene of.
				layer_idx (int): Index in the Pynet of the layer to write the initialization genes of.
			Returns
				str: gene
			"""
			init_fcn_gene = list(Initialization.DNA.genes.keys())[list(Initialization.DNA.genes.values()).index(net.layers[layer_idx].initFcn)]
			return npe.Int2Bin(init_fcn_gene,Initialization.DNA.init_fcn_bits)

		def Read_Gene(genes: str,next_gene_starts_idx: int):
			"""Reads the net gene binary string.
			Args:
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the layer initialization allele begins in the genes.
			Returns
				str: initFcn
				int: next_gene_starts_idx
			"""
			next_gene_ends_idx = next_gene_starts_idx + Initialization.DNA.init_fcn_bits
			initFcn = Initialization.DNA.genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
			next_gene_starts_idx = next_gene_ends_idx
			return initFcn,next_gene_starts_idx

class Transfer:
	"""For calculating the signal transfer between layers.
	Notes:
		'compete' transfer function not utilized yet because may not be used with gradient descent as it has no gradient?
	TODO:
		Add poslin, satlin, poslin, ramp, tribas satlins, radbas, radbasn, netinv, hardlim, hardlims, elliotsig, wavelet
		https://www.hindawi.com/journals/tswj/2013/632437/
		(u-t)/g ?*1/sqrt(g)
		https://github.com/nickgeoca/cwt-tensorflow
	"""

	def Tansig(N_tf: tf.Tensor):
		"""Hyperbolic tangent sigmoid transfer function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/tansig.html
		Notes:
			N[n_members,n_samples,n_nodes] => A[n_members,n_samples,n_nodes]
		"""
		#tansig(n) = 2/(1+exp(-2*n))-1
		#return tf.Print(A_tf, [A_tf], message="--- A_tf:\n")
		with tf.name_scope('Tansig'): return tf.subtract((2 / (1 + tf.exp(-2 * N_tf))),1)

	def Logsig(N_tf: tf.Tensor):
		"""Logarithmic sigmoid transfer function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/logsig.html
		"""
		#logsig(n) = 1 / (1 + exp(-n))
		with tf.name_scope('Logsig'): return 1 / (1 + tf.exp(-N_tf))

	def Purelin(N_tf: tf.Tensor):
		"""Purely linear transfer function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/purelin.html
		"""
		#return tf.Print(A_tf, [A_tf], message="--- A_tf:\n")
		with tf.name_scope('Purelin'): return N_tf

	def Compete(N_tf: tf.Tensor):
		"""Competitive transfer function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/compet.html
		Notes:
			Cannot use with gradient descent as it is discontinuous and not differentiable.
		"""
		#return tf.Print(A_tf, [A_tf], message="--- A_tf:\n")
		with tf.name_scope('Compete'): return tf.cast(tf.equal(N_tf, tf.reshape(tf.reduce_max(N_tf, axis=1), (-1, 1))),tf.float32)

	def Softmax(N_tf: tf.Tensor,dim: int):
		"""Normalized exponential transfer function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
			dim (int): Dimension of the dataset samples.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/softmax.html
			https://en.wikipedia.org/wiki/Softmax_function
		Notes:
			Usually the final layer in a classification Pynet.
		"""
		#exp(n)/sum(exp(n))
		#N[] => A[n_samples,n_classes]
		#return tf.exp(N_tf) / tf.reduce_sum(tf.exp(N_tf), axis=dim)
		#return tf.nn.softmax(N_tf,axis=dim,name='Softmax')
		with tf.name_scope('Softmax'): return tf.nn.softmax(N_tf, axis=dim)

	def Softplus(N_tf: tf.Tensor):
		"""Softplus rectified transfer function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://sefiks.com/2017/08/11/softplus-as-a-neural-networks-activation-function/
			https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
		"""
		#softplus(n) = log(1 + exp(n))
		with tf.name_scope('Softplus'): return tf.log(1 + tf.exp(N_tf))

	def ReLu(N_tf: tf.Tensor):
		"""Rectified exponential linear unit activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu
		"""
		with tf.name_scope('ReLu'): return tf.nn.relu(N_tf)

	def eLu(N_tf: tf.Tensor):
		"""Exponential linear unit activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://arxiv.org/abs/1511.07289
		"""
		with tf.name_scope('eLu'): return tf.nn.elu(N_tf)

	def SeLu(N_tf: tf.Tensor):
		"""Scaled exponential linear unit activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://arxiv.org/abs/1706.02515
		"""
		with tf.name_scope('SeLu'): return tf.nn.selu(N_tf)

	def Swish(N_tf: tf.Tensor):
		"""Swish activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://arxiv.org/abs/1710.05941
		"""
		with tf.name_scope('Swish'): return tf.nn.swish(N_tf)

	def SoftSign(N_tf: tf.Tensor):
		"""Soft sign activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/nn/softsign
		"""
		with tf.name_scope('SoftSign'): return tf.nn.softsign(N_tf)

	def LogSoftmax(N_tf: tf.Tensor):
		"""Logarythmic softmax activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax
		"""
		with tf.name_scope('LogSoftmax'): return tf.nn.log_softmax(N_tf)

	def LeakyReLu(N_tf: tf.Tensor):
		"""Leaky rectified linear unit activation function.
		Args:
			N_tf (tf.Tensor): Neuron input signal.
		Returns:
			tf.Tensor: A_tf
		Ref:
			https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
		"""
		with tf.name_scope('LeakyReLu'): return tf.nn.leaky_relu(N_tf)

	class DNA:

		fcns = {
			'tansig': lambda x,fdim: Transfer.Tansig(x),
			'logsig': lambda x,fdim: Transfer.Logsig(x),
			'purelin': lambda x,fdim: Transfer.Purelin(x),
			#'compete': lambda x,fdim: Transfer.Compete(x),
			'softmax': lambda x,fdim: Transfer.Softmax(x,fdim),
			'softplus': lambda x,fdim: Transfer.Softplus(x),
			'relu': lambda x,fdim: Transfer.ReLu(x),
			'elu': lambda x,fdim: Transfer.eLu(x),
			'selu': lambda x,fdim: Transfer.SeLu(x),
			'swish': lambda x,fdim: Transfer.Swish(x),
			'softsign': lambda x,fdim: Transfer.SoftSign(x),
			'logsoftmax': lambda x,fdim: Transfer.LogSoftmax(x),
			'leakyrelu': lambda x,fdim: Transfer.LeakyReLu(x),
			}
		genes = {
			0: 'tansig',
			1: 'logsig',
			2: 'purelin',
			#3: 'compete', #not included, cannot train atm
			3: 'softmax',
			4: 'softplus',
			5: 'relu',
			6: 'elu',
			7: 'selu',
			8: 'swish',
			9: 'softsign',
			10: 'logsoftmax',
			11: 'leakyrelu',
			}

		input_range = {
			'tansig': [-2,2],
			'logsig': [-4,4],
			'purelin': [-float('Inf'),float('Inf')],
			#'compete': [-float('Inf'),float('Inf')],
			'softmax': [-float('Inf'),float('Inf')],
			'softplus': [-float('Inf'),float('Inf')],
			'relu': [-float('Inf'),float('Inf')],
			'elu': [-float('Inf'),float('Inf')],
			'selu': [-float('Inf'),float('Inf')],
			'swish': [-float('Inf'),float('Inf')],
			'softsign': [-float('Inf'),float('Inf')],
			'logsoftmax': [-float('Inf'),float('Inf')],
			'leakyrelu': [-float('Inf'),float('Inf')],
			}

		output_range = {
			'tansig': [-1,1],
			'logsig': [0,1],
			'purelin': [-float('Inf'),float('Inf')],
			#'compete': [0,1],
			'softmax': [0,1],
			'softplus': [0,float('Inf')],
			'relu': [0,float('Inf')],
			'elu': [0,float('Inf')],
			'selu': [0,float('Inf')],
			'swish': [-float('Inf'),float('Inf')],
			'softsign': [0,float('Inf')],
			'logsoftmax': [-float('Inf'),float('Inf')],
			'leakyrelu': [-float('Inf'),float('Inf')],
			}
	
		transfer_fcn_bits = 8

		def Write_Gene(net: Pynet,layer_idx: int):
			"""Writes the pynet layer's Transfer Function as a binary string.
			Args:
				net (Pynet): Pynet to write the layer transfer gene of.
				layer_idx (int): Index in the Pynet of the layer to write the transfer genes of.
			Returns
				str: gene
			"""
			
			transfer_fcn_gene = list(Transfer.DNA.genes.keys())[list(Transfer.DNA.genes.values()).index(net.layers[layer_idx].transferFcn)]
			return npe.Int2Bin(transfer_fcn_gene,Transfer.DNA.transfer_fcn_bits)

		def Read_Gene(genes: str,next_gene_starts_idx: int):
			"""Reads the pynet's gene binary string.
			Args:
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the layer initialization allele begins in the genes.
			Returns:
				str: transferFcn
				int: next_gene_starts_idx
			"""

			next_gene_ends_idx = next_gene_starts_idx + Transfer.DNA.transfer_fcn_bits
			transferFcn = Transfer.DNA.genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
			next_gene_starts_idx = next_gene_ends_idx
			return transferFcn,next_gene_starts_idx

class PreProcess:
	"""Processing of data before signal input to the Pynet.
	Args:
		settings (processSettings): Specific settings for process used.
		processFcn (str): 'mapminmax' | 'standardize'
	Ref:
		https://www.mathworks.com/help/deeplearning/ug/choose-neural-network-input-output-processing-functions.html
	TODO:
		Add Principle Component Analysis
		Add 2d to 1d mapping -> https://en.wikipedia.org/wiki/Hilbert_curve
	"""

	def __init__(self,kwargs):
			
		self.settings = kwargs['settings']
		self.processFcn = kwargs['processFcn']

		return super().__init__()

	def Mapminmax(X_tf: tf.Tensor,xmax_tf: tf.Tensor,xmin_tf: tf.Tensor,ymax_tf: tf.Tensor,ymin_tf: tf.Tensor):
		"""Normalize inputs to fall in the range [−1, 1]
		Args:
			X_tf (tf.Tensor): Input dataset.
			xmax_tf (tf.Tensor): Maximum values input dataset may take as a Tensor.
			xmin_tf (tf.Tensor): Minimum alues input dataset may take as a Tensor.
			ymax_tf (tf.Tensor): Maximum values output dataset may take as a Tensor.
			ymin_tf (tf.Tensor): Minimum values output dataset may take as a Tensor.
		Returns:
			tf.Tensor: Xp_tf
		Ref:
			https://www.mathworks.com/help/nnet/ref/mapminmax.html
		Notes:
			X[n_members,n_samples,n_features] => Xp[n_members,n_samples,n_features]
		"""
		with tf.name_scope('Mapminmax'):
			yR_tf = tf.subtract(ymax_tf,ymin_tf,name='Y_range')
			xR_tf = tf.subtract(xmax_tf,xmin_tf,name='X_range')
			xpR_tf = tf.subtract(X_tf,xmin_tf,name='Xp_range')
			Xp_tf = tf.add((yR_tf * xpR_tf) / xR_tf, ymin_tf)
			
			return Xp_tf

	def Standardize(X_tf: tf.Tensor, xmean_tf: tf.Tensor, xstd_tf: tf.Tensor):
		"""Standardize data also known as Z-score normalization.
		Args:
			X_tf (tf.Tensor): Input dataset.
			xmean_tf (tf.Tensor): Mean value of input dataset as a Tensor.
			xstd_tf (tf.Tensor): Standard deviation of input dataset as a Tensor.
		Returns:
			tf.Tensor: Xp_tf
		Ref:
			http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py
		"""

		#mean, var = tf.nn.moments(X_tf,0) #TODO: which axis?

		#TODO: if input is only 1 value, this will be nan?
		#X_tf = tf.Print(X_tf, [xmean_tf], message="--- mean:\n")
		#X_tf = tf.Print(X_tf, [xstd_tf], message="--- std:\n")

		#TODO: maybe add epsilon to xstd ? - because divide by zero?
		#std of 0 means the values are the same, x - mean will be 0, 0/0
		#z score will be defined as 0 because its supposed to be a comparison scale

		return ((X_tf - xmean_tf) / xstd_tf)

	def PrincipleComponentAnalysis(X_tf: tf.Tensor):#, xmean_tf: tf.Tensor,ss_tf: tf.Tensor,us_tf: tf.Tensor,vs_tf: tf.Tensor):
		"""Compute PCA on the bottom two dimensions of x, eg assuming dims = [..., observations, features]
		Args:
			INCOMPLETE
			X_tf (tf.Tensor): Inputs
			xmean_tf (tf.Tensor):
		Returns:
			?X?X?
		Ref:
			https://ewanlee.github.io/2018/01/17/PCA-With-Tensorflow/
			https://gist.github.com/N-McA/bbbaed9d1a4b7c316f5d28cef1b96bdd
			https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
			https://plot.ly/ipython-notebooks/principal-component-analysis/
		"""
		
		#return tft.pca(X_tf,2,tf.float32)

		raise ValueError('PCA incomplete')
		#TODO: these settings should be stored from training set, like the center
		#basically, 

		# Center
		X_tf -= tf.reduce_mean(X_tf, -2, keepdims=True)

		# Currently, the GPU implementation of SVD is awful.
		# It is slower than moving data back to CPU to SVD there
		# https://github.com/tensorflow/tensorflow/issues/13222

		with tf.device('/cpu:0'):
			ss, us, vs = tf.svd(X_tf, full_matrices=False, compute_uv=True)

		ss = tf.expand_dims(ss, -2)
		projected_data = us * ss

		# Selection of sign of axes is arbitrary.
		# This replicates sklearn's PCA by duplicating flip_svd
		# https://github.com/scikit-learn/scikit-learn/blob/7ee8f97e94044e28d4ba5c0299e5544b4331fd22/sklearn/utils/extmath.py#L499
		r = projected_data
		abs_r = tf.abs(r)
		m = tf.equal(abs_r, tf.reduce_max(abs_r, axis=-2, keepdims=True))
		signs = tf.sign(tf.reduce_sum(r * tf.cast(m, r.dtype), axis=-2, keepdims=True))
		result = r * signs

		return result

	class DNA:

		fcns = {
			'mapminmax': lambda x,xmax,xmin,ymax,ymin,xmean,xstd: PreProcess.Mapminmax(x,xmax,xmin,ymax,ymin),
			'standardize': lambda x,xmax,xmin,ymax,ymin,xmean,xstd: PreProcess.Standardize(x,xmean,xstd),
			#'pca': lambda x,xmax,xmin,ymax,ymin,xmean,xstd: PreProcess.PrincipleComponentAnalysis(x),
			}
		genes = {
			0: 'mapminmax',
			1: 'standardize',
			#2: 'pca',
			}
		
		pre_process_bits = 8
		
		def Write_Gene(net: Pynet):
			"""Writes the pynet's preprocess as a binary string.
			Args:
				net (Pynet): Pynet to write the preprocess gene of.
			Returns
				str: gene
			"""
			preproc_gene = list(PreProcess.DNA.genes.keys())[list(PreProcess.DNA.genes.values()).index(net.preProcessor.processFcn)]
			return npe.Int2Bin(preproc_gene,PreProcess.DNA.pre_process_bits)

		def Read_Gene(genes: str,next_gene_starts_idx: int):
			"""Reads the pynet's preprocess gene binary string.
			Args:
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the preprocess allele begins in the genes.
			Returns:
				str: transferFcn
				int: next_gene_starts_idx
			"""
			next_gene_ends_idx = next_gene_starts_idx + PreProcess.DNA.pre_process_bits
			processFcn = PreProcess.DNA.genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
			next_gene_starts_idx = next_gene_ends_idx
			return processFcn,next_gene_starts_idx

class processSettings:
	"""Settings for PreProcess functions
	Args:
		xmax (float): Maximum values input dataset may take.
		xmin (float): Minimum alues input dataset may take.
		ymax (float): Maximum values output dataset may take.
		ymin (float): Minimum values output dataset may take.
		xmean (float): Mean value of input dataset.
		xstd (float): Standard deviation of input dataset.
	"""
	def __init__(self,kwargs):

		self.xmax = kwargs['xmax']
		self.xmin = kwargs['xmin']
		self.ymax = kwargs['ymax']
		self.ymin = kwargs['ymin']
		self.xmean = kwargs['xmean']
		self.xstd = kwargs['xstd']

		return super().__init__()

class Performance:
	"""Performance evaluation functions for assessing Pynet training session.
	TODO: 
		Add msesparse?
	"""

	def CrossEntropy(T_tf: tf.Tensor,y_tf: tf.Tensor):
		"""Computes the cross entropy loss of a classification network
		Args:
			T_tf (tf.Tensor): Target dataset
			y_tf (tf.Tensor): Simulated signal output dataset
		Returns:
			tf.Tensor: loss
		Ref:
			https://www.mathworks.com/help/nnet/ref/crossentropy.html
		"""

		scaled_signal = (y_tf - tf.reduce_max(y_tf))
		normalized_signal = scaled_signal - tf.reduce_logsumexp(scaled_signal)
		
		with tf.name_scope('CrossEntropy'):
			return -tf.reduce_sum(T_tf * normalized_signal)

	def MSE(T_tf: tf.Tensor,y_tf: tf.Tensor):
		"""Computes the mean squared error between the target and simulation.
		Args:
			T_tf (tf.Tensor): Target dataset
			y_tf (tf.Tensor): Simulated signal output dataset
		Returns:
			tf.Tensor: loss
		Ref:
			https://www.mathworks.com/help/nnet/ref/mse.html
		"""
		with tf.name_scope('MSE'):
			return tf.reduce_mean(tf.reduce_mean(tf.square(T_tf - y_tf),axis=1))

	def MAE(T_tf: tf.Tensor,y_tf: tf.Tensor):
		"""Computes mean absolute error between the target and simulation.
		Args:
			T_tf (tf.Tensor): Target dataset
			y_tf (tf.Tensor): Simulated signal output dataset
		Returns:
			tf.Tensor: loss
		Ref:
			https://www.mathworks.com/help/nnet/ref/mae.html
		"""
		with tf.name_scope('MAE'):
			return tf.reduce_mean(tf.abs(y_tf - T_tf))
		
	def SAE(T_tf: tf.Tensor,y_tf: tf.Tensor):
		"""Computes sum absolute error between the target and simulation.
		Args:
			T_tf (tf.Tensor): Target dataset
			y_tf (tf.Tensor): Simulated signal output dataset
		Returns:
			tf.Tensor: loss
		Ref:
			https://www.mathworks.com/help/nnet/ref/sae.html
		"""
		with tf.name_scope('SAE'):
			return tf.reduce_sum(tf.abs(y_tf - T_tf))

	def SSE(T_tf: tf.Tensor,y_tf: tf.Tensor):
		"""Computes sum squared error between the target and simulation.
		Args:
			T_tf (tf.Tensor): Target dataset
			y_tf (tf.Tensor): Simulated signal output dataset
		Returns:
			tf.Tensor: loss
		Ref:
			https://www.mathworks.com/help/nnet/ref/sse.html
		"""
		with tf.name_scope('SSE'):
			return tf.reduce_sum(tf.squared_difference(y_tf, T_tf))

	class DNA:

		fcns = {
		'crossentropy': lambda t,y: Performance.CrossEntropy(t,y),
		'mse': lambda t,y: Performance.MSE(t,y),
		'mae': lambda t,y: Performance.MAE(t,y),
		'sse': lambda t,y: Performance.SSE(t,y),
		'sae': lambda t,y: Performance.SAE(t,y)
		}

		genes = {
			0: 'crossentropy',
			1: 'mse',
			2: 'mae',
			3: 'sse',
			4: 'sae'
			}

		perform_fcn_bits = 8

		def Write_Gene(net: Pynet):
			"""Writes the pynet's Performance Function as a binary string.
			Args:
				net (Pynet): Pynet to write the performance gene of.
			Returns:
				str: gene
			"""

			perf_gene = list(Performance.DNA.genes.keys())[list(Performance.DNA.genes.values()).index(net.trainer.performFcn)]
			return npe.Int2Bin(perf_gene,Performance.DNA.perform_fcn_bits)

		def Read_Gene(genes: str,next_gene_starts_idx: int):
			"""Reads the pynet's gene binary string.
			Args:
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the performance allele begins in the genes.
			Returns:
				str: performFcn
				int: next_gene_starts_idx
			"""

			next_gene_ends_idx = next_gene_starts_idx + Performance.DNA.perform_fcn_bits
			performFcn = Performance.DNA.genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
			next_gene_starts_idx = next_gene_ends_idx
			return performFcn,next_gene_starts_idx

class DatasetDivider:
	"""Class for dividing a dataset.
	Notes:
		divideParams_bits ->  max 127n need 100, must be divisible by 3 or will screw up reading/writing dna
	"""
	
	def Format_Percentages(tp: float,vp: float,sp: float):
		"""Formats percentage ratios to be floats < 1.0.
		Args:
			tp (float): Training percentage
			vp (float): Validation percentage
			sp (float): Testing percentage
		Returns:
			float: tp
			float: vp
			float: sp
		Notes:
			Percentages can be such as 60.5 which must be formatted to 0.605
		"""

		#check if add up to 1, not already correctly formatted
		if np.round(tp + vp + sp,5) != float(1): #round due to quantiization

			#if dont add up to 1, check if add up to 100
			if (tp + vp + sp) == int(100):

				#add up to 100, divide to get correct float format
				tp = tp/100
				vp = vp/100
				sp = sp/100
			else:
				#neither add up to 1.0 or 100, incorrect
				raise ValueError('DatasetDivider.Format_Percentages: dataset division percentages/ratios must either add up to 1.0 or 100')

		return tp,vp,sp

	def DivideRandom(X: np.array,T: np.array,tp: float,vp: float,sp: float):
		"""Divide X,T data randomly into training, validation, and test sets
		Args:
			X (np.array): Input dataset
			T (np.array): Targets dataset
			tp (float): Training percentage
			vp (float): Validation percentage
			sp (float): Testing percentage
		Returns:
			np.array: Xt
			np.array: Xv
			np.array: Xs
			np.array: Tt
			np.array: Tv
			np.array: Ts
		"""
			
		#percentage check
		tp,vp,sp = DatasetDivider.Format_Percentages(tp,vp,sp)
			
		#split training out in random manner
		Xt, Xs, Tt, Ts = train_test_split(X, T, test_size=(1-tp))

		#split validation and test from remainder in random manner
		Xs, Xv, Ts, Tv = train_test_split(Xs, Ts, test_size=(vp/(1-tp)))

		return Xt,Xv,Xs,Tt,Tv,Ts

	def DivideIndices(X: np.array,T: np.array,tp: float,vp: float,sp: float):
		"""Divide X,T data by indices into training, validation, and test sets.
		Args:
			X (np.array): Input dataset
			T (np.array): Targets dataset
			tp (float): Training percentage
			vp (float): Validation percentage
			sp (float): Testing percentage
		Returns:
			np.array: Xt
			np.array: Xv
			np.array: Xs
			np.array: Tt
			np.array: Tv
			np.array: Ts
		"""

		#percentage check
		tp,vp,sp = DatasetDivider.Format_Percentages(tp,vp,sp)

		#get total number of samples
		n_samples = np.shape(X)[0]

		#split in forward manner based on percentage
		tp_ind = int(np.round(tp * n_samples))
		vp_ind = int(tp_ind + np.round(vp * n_samples))
			
		Xt = X[0:tp_ind]
		Xv = X[tp_ind:vp_ind]
		Xs = X[vp_ind:]

		Tt = T[0:tp_ind]
		Tv = T[tp_ind:vp_ind]
		Ts = T[vp_ind:]

		return Xt,Xv,Xs,Tt,Tv,Ts

	class DNA:

		fcns = {
			'dividerand': lambda X,T,tp,vp,sp: DatasetDivider.DivideRandom(X,T,tp,vp,sp),
			'divideind': lambda X,T,tp,vp,sp: DatasetDivider.DivideIndices(X,T,tp,vp,sp)
			}

		genes = {
			0: 'dividerand',
			1: 'divideind'
			}

		divideFcn_bits = 8
		divideParams_bits = 21 #/3 = 7, max 127n need 100, must be divisible by 3 or will screw up reading/writing dna

		def Write_Gene(divideFcn: str,divideParams: dict):
			"""Writes the Divide Function and Params as a binary string.
			Args:
				divideFcn (str): 'dividerand' | 'divideind'
				divideParams (dict): Contains 'trainRatio','valRatio','testRatio'
			Returns
				str: gene
			"""

			#divide fcn gene
			divideFcn_gene = list(DatasetDivider.DNA.genes.keys())[list(DatasetDivider.DNA.genes.values()).index(divideFcn)]
			divideFcn_gene = npe.Int2Bin(divideFcn_gene,DatasetDivider.DNA.divideFcn_bits)

			#divide param genes
			tp = divideParams['trainRatio']
			vp = divideParams['valRatio']
			sp = divideParams['testRatio']
			tp,vp,sp = DatasetDivider.Format_Percentages(tp,vp,sp) #format to float
			train_gene = divideFcn_gene + npe.Int2Bin(int(divideParams['trainRatio']*100),int(DatasetDivider.DNA.divideParams_bits / 3))
			val_gene = train_gene + npe.Int2Bin(int(divideParams['valRatio']*100),int(DatasetDivider.DNA.divideParams_bits / 3))
			test_gene = val_gene + npe.Int2Bin(int(divideParams['testRatio']*100),int(DatasetDivider.DNA.divideParams_bits / 3))

			return test_gene

		def Read_Gene(genes: str,next_gene_starts_idx: int):
			"""Reads the gene binary string.
			Args:
				genes (str): Binary string representing a configuration.
				next_gene_starts_idx (int): Index where the datasetdivider allele begins in the genes.
			Returns
				str: divideFcn
				dict: divideParams
				int: next_gene_starts_idx
			"""

			next_gene_ends_idx = next_gene_starts_idx + DatasetDivider.DNA.divideFcn_bits
			divideFcn = DatasetDivider.DNA.genes[npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])]
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + int(DatasetDivider.DNA.divideParams_bits / 3)
			tp = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
			next_gene_starts_idx = next_gene_ends_idx

			next_gene_ends_idx = next_gene_starts_idx + int(DatasetDivider.DNA.divideParams_bits / 3)
			vp = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
			next_gene_starts_idx = next_gene_ends_idx
			
			next_gene_ends_idx = next_gene_starts_idx + int(DatasetDivider.DNA.divideParams_bits / 3)
			sp = npe.Bin2Int(genes[next_gene_starts_idx:next_gene_ends_idx])
			next_gene_starts_idx = next_gene_ends_idx
			
			tp,vp,sp = DatasetDivider.Format_Percentages(tp,vp,sp) #format to float

			divideParams = {}
			divideParams['trainRatio'] = tp
			divideParams['valRatio'] = vp
			divideParams['testRatio'] = sp
			
			return divideFcn,divideParams,next_gene_starts_idx

class Tests:

	def main():

		Tests.test_datasetdivider()
		Tests.dna_inport_export_works()
		Tests.test_dna_forming_and_extracting()


		#turned off because they may take substantial computation time
		Tests.test_classification()
		#Tests.test_classification_evolution()


		pass

	def test_classification():

		X,T = Tests.Utils.import_classification_data('iris')

		patternnet = Pynet.Models.PatternnetGroup(8,True)
		#patternnet = Pynet.Models.FitnetGroup(8,True)

		patternnet.ConfigureGraph(showSteps=True)

		#patternnet.trainer.epochs = 50
		#patternnet.layers[0].transferFcn = 'leakyrelu'
		#patternnet.preProcessor.processFcn = 'pca'
		
		training_res = patternnet.Train(X,T)

		err = patternnet.Loss(X,T)

		Y = patternnet.Sim(X,recordGraph=True)

		pass

	def test_classification_evolution():

		X,T = Tests.Utils.import_classification_data('iris')

		#patternnet = Pynet.Models.PatternnetGroup(8,True)
		#patternnet.ConfigureGraph(True)
		#training_res = patternnet.Train(X,T,False)

		evolved_pynet = Pynet.Evolve(X,T,True,10,'brier')

		pass

	def test_datasetdivider():

		tp = 60
		vp = 25
		sp = 15

		correct_tp = 0.60
		correct_vp = 0.25
		correct_sp = 0.15

		tp,vp,sp = DatasetDivider.Format_Percentages(tp,vp,sp)

		if not isinstance(tp,float): raise ValueError('DatasetDivider.Format_Percentages returns incorrect type')
		if not isinstance(vp,float): raise ValueError('DatasetDivider.Format_Percentages returns incorrect type')
		if not isinstance(sp,float): raise ValueError('DatasetDivider.Format_Percentages returns incorrect type')

		if tp != correct_tp: raise ValueError('DatasetDivider.Format_Percentages returns incorrect')
		if vp != correct_vp: raise ValueError('DatasetDivider.Format_Percentages returns incorrect')
		if sp != correct_sp: raise ValueError('DatasetDivider.Format_Percentages returns incorrect')

		#TODO: test dividerandom and indices
		#how to check tho?
		#need big X,T and just check lengths?

		divideParams = {}
		divideParams['trainRatio'] = tp
		divideParams['valRatio'] = vp
		divideParams['testRatio'] = sp

		correct_genes_r = '00000000011110000110010001111'
		correct_genes_i = '00000001011110000110010001111'

		genes_r = DatasetDivider.DNA.Write_Gene('dividerand',divideParams)
		genes_i = DatasetDivider.DNA.Write_Gene('divideind',divideParams)

		if not isinstance(genes_r,str): raise ValueError('DatasetDivider.Write_Gene returns incorrect type')
		if not isinstance(genes_i,str): raise ValueError('DatasetDivider.Write_Gene returns incorrect type')
		if genes_r != correct_genes_r: raise ValueError('DatasetDivider.Write_Gene returns incorrect')
		if genes_i != correct_genes_i: raise ValueError('DatasetDivider.Write_Gene returns incorrect')
		
		divideFcn_r,divideParams_r,i_r = DatasetDivider.DNA.Read_Gene(genes_r,0)
		divideFcn_i,divideParams_i,i_i = DatasetDivider.DNA.Read_Gene(genes_i,0)

		if not isinstance(divideFcn_r,str): raise ValueError('DatasetDivider.Read_Gene returns incorrect type')
		if not isinstance(divideFcn_i,str): raise ValueError('DatasetDivider.Read_Gene returns incorrect type')
		if not isinstance(divideParams_r,dict): raise ValueError('DatasetDivider.Read_Gene returns incorrect type')
		if not isinstance(divideParams_i,dict): raise ValueError('DatasetDivider.Read_Gene returns incorrect type')
		if not isinstance(i_r,int): raise ValueError('DatasetDivider.Read_Gene returns incorrect type')
		if not isinstance(i_i,int): raise ValueError('DatasetDivider.Read_Gene returns incorrect type')
		if divideFcn_r != 'dividerand': raise ValueError('DatasetDivider.Read_Gene returns incorrect')
		if divideFcn_i != 'divideind': raise ValueError('DatasetDivider.Read_Gene returns incorrect')
		if divideParams_r != divideParams: raise ValueError('DatasetDivider.Read_Gene returns incorrect')
		if divideParams_i != divideParams: raise ValueError('DatasetDivider.Read_Gene returns incorrect')
		if i_r != (DatasetDivider.DNA.divideFcn_bits + DatasetDivider.DNA.divideParams_bits): raise ValueError('DatasetDivider.Read_Gene returns incorrect')
		if i_i != (DatasetDivider.DNA.divideFcn_bits + DatasetDivider.DNA.divideParams_bits): raise ValueError('DatasetDivider.Read_Gene returns incorrect')

		pass

	def test_performance():

		pass

	def test_dna_forming_and_extracting():

		dna1 = 75325223835348706574467301109289672659972633489591305814144
		dna2 = 75325223835348706574467301109289672659972633489589158330496
		if (dna1 == dna2): raise ValueError('dna cycle broken')

		net1 = Pynet.DNA.Form(dna1,8,False,3)
		dna1_2 = Pynet.DNA.Extract(net1)
		if (dna1_2 != dna1): raise ValueError('dna cycle broken')

		net2 = Pynet.DNA.Form(dna2,8,False,3)
		dna2_2 = Pynet.DNA.Extract(net2)
		if (dna2_2 != dna2): raise ValueError('dna cycle broken')

		pass

	def dna_inport_export_works():

		#load pynetgroup that is a patternnet
		net = Pynet.Models.PatternnetGroup(7,False)

		#TODO: verify is patternnet?

		#extract dna, reform, and reextract the dna
		dna = Pynet.DNA.Extract(net)
		formed_net = Pynet.DNA.Form(dna,9,False)
		formed_dna = Pynet.DNA.Extract(formed_net)
		if (dna != formed_dna): raise ValueError('dna cycle broken')

		fields = list(net.Dictify().keys())
		net1 = net.Dictify()
		net2 = formed_net.Dictify()
		fields_checked = {}
		ignored_fields = ['IW','LW','b']
		for field in fields:
			if field == 'inputs':
				#TODO: why use idx 0? cause dictify makes them lists but 0 is only idx nesc
				fields_checked[field] = (net1['inputs'][0]['processFcns'] == net2['inputs'][0]['processFcns'])
			else:
				fields_checked[field] = (net1[field] == net2[field])
			if ((not fields_checked[field]) and (field not in ignored_fields)):
				raise ValueError('some fields didnt check out')

		pass

	class Utils:
		def import_classification_data(dataset: str):
			"""Import datasets specifically for testing classifications.
			Args:
				dataset (str): 'iris' |
			Returns:
				np.array: X
				np.array: T
			"""
			#https://archive.ics.uci.edu/ml/datasets/Iris
			#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

			dataset_router = {
				'iris': __file__.split('pynet.py')[0] + './data/irisClassification.json'
				}

			jsonLines = open(dataset_router[dataset],'r').readlines()
			jsonStr = ''.join(jsonLines).replace('\'','')

			dataset = {}
			dataset = json.loads(jsonStr)

			X = np.squeeze(np.transpose(np.array([dataset['X']],np.float32),[2,1,0]),axis=2)
			T = np.squeeze(np.transpose(np.array([dataset['T']],np.float32),[2,1,0]),axis=2)

			return X,T

		def import_regression_data(dataset: str):
			pass
		def import_sequential_data(dataset: str):
			pass


readme = """
# Pynet
#### Simple neural network toolbox specifically designed to train networks simultaneously for evolution optimization.

# Installation
Installing to use in your own scripts in a virtual environment?

`pip install git+https://github.com/pmp47/Pynet`

Installing to edit this code and contribute? Clone/download this repo and...

`pip install -r requirements.txt`


Want to use a GPU? It's recommended to use a more robust package installer...

`conda install tensorflow-gpu`


# Requirements/Features

Pynets have been intentially designed to have some unique features:
* Easily implimented on CPU or GPU.
* Able to train multiple networks simultaneously.
* Modular, so new developments can be easily tested and applied such as new transfer functions.
* Abstract optimization so more focus is on quality of data, rather than exact form of the network.

# Motivation
The main motivation for developing this toolbox was to demonstrate how simple neural networks are to utilize and impliment. Other toolboxes out there are more complicated and less modular.
* Tensorflow is quite popular and provides a great foundation for achieving the first requirement.
* Pynets are groups of identically structured networks which satisfies the second requirement.
* The class division of the categories of Pynets and the wrapping around Tensorflow satisifes the modularity requirement.
* There was no optimization toolbox out there to apply to hyper-parameters like those of an entire neural network so the evolution script provides this ability.

# Usage
##### The simplest way to use a Pynet:
```python
from pynet import Pynet, Tests

#load test classification data such as the classic iris set
X,T = Tests.Utils.import_classification_data('iris')

#set to True to use GPU hardware, usually orders of magnitude faster than CPU
useGPU = False

#number of copies of the neural network in the group -> affects vram reqs and speed
groupSize = 8

#create a simple Pynet group of a Patternnet neural network
nnet = Pynet.Models.PatternnetGroup(groupSize,useGPU)

#configure this Pynet's graph
nnet.ConfigureGraph()

#train the pynet on the input/target datasets
nnet.Train(X,T)

#simulate the output of the network given input data
Y = nnet.Sim(X)

```
##### Evolution Optimization:
Don't know which model to use? Or none of them really get the job done? A pynet may have its structure optimized through an evolutary process. This means you only need to control computing resources rather than the neural network structure.
```python
#capcity of evolving population -> higher takes longer to progress through generations
pop_cap = 25

#method used to evaluate the fitness of each pynet
fitness_method = 'acc'

#create an evolution environment and produce an evolved Pynet
evolved_pynet = Pynet.Evolve(X,T,useGPU,pop_cap,fitness_method,time_limit_minutes=60)

#configure this Pynet's graph
evolved_pynet.ConfigureGraph()

#train this evolved pynet on the dataset
training_result = evolved_pynet.Train(X,T)

#simulate the output of the network given input data
Y = evolved_pynet.Sim(X)
```

# DNA
How is a Pynet able to evolve exactly? It begins with the idea that representing an object with "DNA" is simply an abstraction of how a computer holds data. It means that zeros and ones store the information in a structured pattern. This pattern may be applied in order to represent every single permutation the object may exist as.
```python
#extract dna from the network
dna = Pynet.DNA.Extract(evolved_pynet)

#re-form the network from the extracted dna
formed_net = Pynet.DNA.Form(dna,groupSize+10,useGPU) #prove groupsize isnt part of DNA

```

# Data Preparation

The quality of Pynet you create is very dependant on the datasets used and the type of problem addressed. 
To prepare these datsets for using with a Pynet you must follow the criteria:
* Must be a <strong>np.array</strong>
* X input dataset has the shape -> [n_samples, n_features]
* T target dataset has the shape -> [n_samples, n_classes]


# Saving/Loading a Pynet
There are fundamentally 2 ways to store a Pynet. Each method has its advantages and disadvantages.
### Using DNA
Saving/loading a pynet using DNA <strong>does not store weights/bias</strong> and other information unique to the members of the group. This means a network group stored/retrieved in this manner will need to be retrained before it may perform.
```python

#save the raw dna as text
with open('evolved_dna.txt','w') as text_file:
    text_file.write(str(dna))

#load a network from dna
dna = None
with open('evolved_dna.txt','r') as text_file:
    dna = int(text_file.read())

#reform the network
formed_net = Pynet.DNA.Form(dna,groupSize,useGPU)

```

### Dictifying
This process transforms the network into a dictionary then into json text. This makes reusing a specific network possible.
```python
import json

#save net as a dict in json format
with open('evolved_net.json','w') as outfile:
    outfile.write(json.dumps(evolved_net.Dictify()))

#load
evolved_net = None_
with open('evolved_net.json','r') as outfile:
    evolved_net = Pynet.IO.Load(outfile.read(),useGPU)

```

# Tips

Changing a pynet's structure should be done before configuration
```python
#change a fitnet
fitnet = Pynet.Models.FitnetGroup(groupSize,useGPU)

#into a cascade
fitnet.isInput[-1] = True

#with a different final transfer
fitnet.layers[-1].transferFcn = 'swish'

#then configure
fitnet.ConfigureGraph()

```
Trained a very large group but only need a single network from it?
```python
#pick a group member to get
member_idx = 7

#get the member out
member = Pynet.MemberOut(nnet,member_idx)

#they can still simulate a signal output
Y = member.Sim(X)
```

# Further Information
Looking for more explanation? The code itself is documented in a way meant to be read and understood easily. A good place to start with this package would be:
```python
pynet.Tests.main
```

# TODO
 Currently there are yet to be implimented features:
 * Complete recurrent layer capability so sequential networks may be evolved, such has LSTM
 * Higher dimensionality of layers must be added in order to support typical computer vision applications such as 2-D Convolutions


 ##### fin

"""
