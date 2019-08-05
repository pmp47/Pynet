#--start requirements--
#pip installs
import numpy as np

#customs

#builtins

#--end requirements--

def find_contains(str_target: str,str_list: list):
	"""Finds where a list of strings contains a specified target string.
	Args:
		str_target (str): Target string to check if list contains
		str_list (list): List of strings
	Returns:
		np.array dtype=int64: Indices where specified target string is contained in the list.
	"""
	return np.flatnonzero(np.core.defchararray.find(str_list,str_target)!=-1)

def Int2Bin(integer: int, n_bits: int):
	"""Turn an integer into a binary string.
	Args:
		integer (int): Integer to convert.
		n_bits (int): Number of bits to represent the integer.
	Returns:
		str: binary_str
	"""
	formatstr = '0' + str(n_bits) + 'b'
	return format(integer,formatstr)

def Bin2Int(binaryStr_or_np_binaryArray: str):
	"""Turn a binary string into an integer.
	Args:
		binaryStr_or_np_binaryArray(str or np.array): A binary string representation of an integer OR a np.array.
	Returns:
		int: integer
	"""
	try:
		return int(binaryStr_or_np_binaryArray,2)
	except:
		return int(binaryStr_or_np_binaryArray.dot(1 << np.arange(binaryStr_or_np_binaryArray.size)[::-1]))

def rolling_std(x: np.array,period: int,use_less_mem=False):
	"""Compute the rolling standard deviation of an array based on a window period.
	Args:
		x (np.array): Array of values to compute rolling standard deviation of.
		period (int): The size of the rolling window to use.
		use_less_mem (bool): NOT YET ->
	Returns:
		std: np.array
	"""

	if use_less_mem:
		raise ValueError('incorrect? has to do with zeros_like vs zeros (shape) because nans?')
		#initialize nan array
		_std = np.zeros_like(x)
		_std[:] = np.NaN

		#standard deviation of rolling window
		for i in range(period-1,len(x)):
			window = x[i-period+1:i+1]
			_std[i] = np.std(window,ddof=1)
	else:
		x_window = np.zeros([period,x.shape[0]])
		x_window[:,:] = np.NaN
		x_window[0,:] = x
		for p in range(1,period):
			x_window[p,p:] = x[:-p]
		_std = np.std(x_window,axis=0)
	return _std

def rolling_sum(x: np.array,period: int):
	"""Compute the rolling sum of an array based on a window period.
	Args:
		x (np.array): Array of values to compute rolling sum of.
		period (int): The size of the rolling window to use.
		use_less_mem (bool): NOT YET ->
	Returns:
		rsum: np.array
	"""
	#apply rolling window sum by staggering arrays and getting difference in cumulative sums

	#initialize nan array
	rsum = np.zeros(x.shape)
	rsum[:] = np.nan

	#compute cumulative sum of input
	cumsum = np.cumsum(np.insert(x, 0, 0)) 

	#get difference
	rsum[period-1:] = (cumsum[period:] - cumsum[:-period])
	
	return rsum

def rolling_avg(x: np.array,period: int):
	"""Compute the rolling average of an array based on a window period.
	Args:
		x (np.array): Array of values to compute rolling average of.
		period (int): The size of the rolling window to use.
		use_less_mem (bool): NOT YET ->
	Returns:
		avg: np.array
	"""
	rsum = rolling_sum(x,period)

	avg = rsum / float(period) #use float to preserve float division

	return avg

def Normalize(normThisData: np.array,toThisData: np.array):
	"""Normalize one dataset to another to produce a unitless output.
	Args:
		normThisData (np.array): 
		toThisData (np.array):
	Returns:
		np.array: normalized_data
	Notes: 
		Currently only works for 1D arrays shape=[m,] ?????true? and returns same shape 1D array
	"""
	#TODO: needs Tests.Normalize
	return (toThisData - normThisData) / toThisData

def GetIntersections(y_vals,y_line,min_idz,fromBelow=True):
	"""Get indices of when y_vals intersect y_line.
	Args:
		y_vals (np.array): Array representing the signal to get intersections of.
		y_line (np.array): Array where each entry is the value in the equation 'y = value'
		min_idz (np.array): Array of where each entry is the minimum index for the corresponding y_line entry.
		fromBelow (bool): Default True to get intersection approaching the y_val function from below.
	Returns:
		np.array: Indices of Intersection
	Notes:
		When a y_line[i] <= y_vals[i] and fromBelow, res[i] = i
		When a y_line[i] >= y_vals[i] and not fromBelow, res[i] = i
		This means if y_line is already past the y_val depending on the fromBelow, the index will be the point this occurs.
	"""

	intersect_idz_res = np.zeros_like(y_line,np.int64)
			
	if fromBelow:
		for i in range(0,y_line.size): #get intersection for each y_line value
			intersect_mask = y_vals >= y_line[i] #intersection when >= because fromBelow
			if np.any(intersect_mask): #if any intersection occurs
				if np.any(intersect_mask[min_idz[i]:]) & (min_idz[i] >= 0): #if intersection occurs at or after min idx
					intersect_idz = np.nonzero(intersect_mask)
					intersect_idz_res[i] = intersect_idz[0][(intersect_idz >= min_idz[i])[0]][0] #get soonest intersection
				else: #no intersection after the min idx, set idz to infinity
					intersect_idz_res[i] = np.iinfo(np.int64).max
			else: #no intersection, set idz to infinity
				intersect_idz_res[i] = np.iinfo(np.int64).max

	else:
		for i in range(0,y_line.size):
			intersect_mask = y_vals <= y_line[i] #intersection when <= because not fromBelow
			if np.any(intersect_mask): #if any intersection occurs
				if np.any(intersect_mask[min_idz[i]:]) & (min_idz[i] >= 0): #if intersection occurs at or after min idx
					intersect_idz = np.nonzero(intersect_mask)
					intersect_idz_res[i] = intersect_idz[0][(intersect_idz >= min_idz[i])[0]][0] #get soonest intersection
				else: #no intersection after the min idx, set idz to -1
					intersect_idz_res[i] = np.iinfo(np.int64).max
			else: #no intersection, set idz to -1
				intersect_idz_res[i] = np.iinfo(np.int64).max

			
	if np.any(intersect_idz_res == 0):
		bad_idz = np.nonzero(intersect_idz_res == 0)[0]
		if bad_idz[0] == i:
			# means the final value did not intersect (because its the final value)
			#so just remove it
			intersect_idz_res[bad_idz[0]] = np.iinfo(np.int64).max
		else:
			raise ValueError('Some intersections were not determined.')

	if np.any((intersect_idz_res >= y_vals.size) & (intersect_idz_res < np.iinfo(np.int64).max)):
		bad_idz = np.nonzero((intersect_idz_res >= y_vals.size) & (intersect_idz_res < np.iinfo(np.int64).max))
		raise ValueError('idx too large?')

	return intersect_idz_res

def GetTrailingIntersections(y_vals,y_line,min_idz,fromBelow=True):
	
	raise ValueError('currently incomplete')
	#TODO: trailing stop intersections

	intersect_idz_res = GetIntersections(y_vals,y_line,min_idz,fromBelow=fromBelow)

	#first mask the intersections that cant possibly have a trail collision because the trail doesnt take affect

	#second ..?
	
	pass

class Tests:

	def main():

		Tests.test_find_contains()
		Tests.test_int2bin()
		Tests.test_bin2int()
		Tests.test_rolling_std()
		Tests.test_rolling_sum()
		Tests.test_rolling_avg()
		Tests.test_normalize()
		Tests.test_getintsersections()

	def test_find_contains():

		str_target = 'find me'
		str_list = ['i am','a list','find me','of strings','find me']
		correct_idx = [2,4]
		indices = find_contains(str_target,str_list)
		if not isinstance(indices,np.ndarray): raise ValueError('find_contains returns incorrect type')
		if indices[0] != correct_idx[0]: raise ValueError('find_contains returns incorrect idx')
		if indices[1] != correct_idx[1]: raise ValueError('find_contains returns incorrect idx')

	def test_int2bin():

		integer = 55
		n_bits = 2
		correct_binary_str = '110111'
		binary_str = Int2Bin(integer,n_bits)
		if not isinstance(binary_str,str): raise ValueError('int2bin returns incorrect type')
		if binary_str != correct_binary_str: raise ValueError('int2bin returns incorrect binary_str')

	def test_bin2int():

		binary_arr = np.array([1,1,0,1,1,1])
		binary_str = '110111'
		correct_integer = 55
		integer_a = Bin2Int(binary_arr)
		integer_s = Bin2Int(binary_str)
		if not isinstance(integer_a,int): raise ValueError('bin2int returns incorrect type')
		if not isinstance(integer_s,int): raise ValueError('bin2int returns incorrect type')
		if integer_a != correct_integer: raise ValueError('bin2int returns incorrect integer')
		if integer_s != correct_integer: raise ValueError('bin2int returns incorrect integer')

	def test_rolling_std():

		x = np.array([5,5,2,8,6,9,5,7,1,2,44,8,5,66])
		period = 4
		correct_std = np.array([2.1213203435596424, 2.165063509461097, 2.680951323690902, 1.5811388300841898, 1.479019945774904, 2.958039891549808, 2.384848003542364, 17.755280904564703, 17.66882848408462, 17.020208576865326, 25.488968201949643])
		std = rolling_std(x,period)
		if not isinstance(std,np.ndarray): raise ValueError('rolling_std returns incorrect type')
		if len(std) != len(x): raise ValueError('rolling_std returns incorrect length with nans')
		std = std[~np.isnan(std)] #np.nan != np.nan -> nans will never be equal
		if np.any(std != correct_std): raise ValueError('rolling_std returns incorrect std')

	def test_rolling_sum():

		x = np.array([5,5,2,8,6,9,5,7,1,2,44,8,5,66])
		period = 4
		correct_sum = np.array([20.0, 21.0, 25.0, 28.0, 27.0, 22.0, 15.0, 54.0, 55.0, 59.0, 123.0])
		rsum = rolling_sum(x,period)
		if not isinstance(rsum,np.ndarray): raise ValueError('rolling_sum returns incorrect type')
		if len(rsum) != len(x): raise ValueError('rolling_sum returns incorrect length with nans')
		rsum = rsum[~np.isnan(rsum)] #np.nan != np.nan -> nans will never be equal
		if np.any(rsum != correct_sum): raise ValueError('rolling_sum returns incorrect sum')

	def test_rolling_avg():

		x = np.array([5,5,2,8,6,9,5,7,1,2,44,8,5,66])
		period = 4
		correct_avg = np.array([5.0, 5.25, 6.25, 7.0, 6.75, 5.5, 3.75, 13.5, 13.75, 14.75, 30.75])
		avg = rolling_avg(x,period)
		if not isinstance(avg,np.ndarray): raise ValueError('rolling_avg returns incorrect type')
		if len(avg) != len(x): raise ValueError('rolling_avg returns incorrect length with nans')
		avg = avg[~np.isnan(avg)] #np.nan != np.nan -> nans will never be equal
		if np.any(avg != correct_avg): raise ValueError('rolling_avg returns incorrect avg')

	def test_normalize():

		pass

	def test_getintsersections():

		y_vals = np.array([2,4,6,8,5,3,1,10])
		y_line = np.array([3,9,10,4,7,12,8,15])
		min_idz = np.array([0,1,2,3,4,5,6,7])
		correct_idz_below = np.array([1,7,7,3,7,np.iinfo(np.int64).max,7,np.iinfo(np.int64).max])
		idz_res_below = GetIntersections(y_vals,y_line,min_idz,fromBelow=True)
		if not isinstance(idz_res_below,np.ndarray): raise ValueError('GetIntersections fromBelow returns incorrect type')
		if len(idz_res_below) != len(y_vals): raise ValueError('GetIntersections fromBelow returns incorrect length')
		if np.any(idz_res_below != correct_idz_below): raise ValueError('GetIntersections fromBelow returns incorrect indices')

		y_vals = np.array([20,50,40,35,20,10,0])
		y_line = np.array([15,30,15,00,25,-20,-30])
		in_idz = np.array([0,1,2,3,4,5,6,7])
		correct_idz_above = np.array([5,4,5,6,4,np.iinfo(np.int64).max,np.iinfo(np.int64).max])
		idz_res_above = GetIntersections(y_vals,y_line,min_idz,fromBelow=False)
		if not isinstance(idz_res_above,np.ndarray): raise ValueError('GetIntersections fromAbove returns incorrect type')
		if len(idz_res_above) != len(y_vals): raise ValueError('GetIntersections fromAbove returns incorrect length')
		if np.any(idz_res_above != correct_idz_above): raise ValueError('GetIntersections fromAbove returns incorrect indices')

		pass