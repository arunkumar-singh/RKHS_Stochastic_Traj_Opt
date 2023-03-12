



from numpy import *



def pol_matrix_comp(t):

		delt = abs(t[1]-t[0])[0]

		num = len(t)

		Ad = array([[1, delt, 0.5*delt**2],[0, 1, delt],[0, 0, 1]])
		Bd = array([[1/6*delt**3, 0.5*delt**2, delt]]).T

		
		P = zeros((num-1, num-1))
	    
		Pdot = zeros((num-1, num-1))
	    
	    
		Pddot = zeros((num-1, num-1))

		Pint =  zeros((num, 3))
		Pdotint =  zeros((num, 3))
		Pddotint = zeros((num, 3))        

		for i in range(0,num-1):
			for j in range(0,i):
				temp = dot(linalg.matrix_power(Ad, (i-j)), Bd)
				
				P[i][j] = temp[0]
				Pdot[i][j] = temp[1]
				Pddot[i][j] = temp[2]

		for i in range(0, num):
			temp = linalg.matrix_power(Ad,i)
			
			Pint[i] = temp[0]
			Pdotint[i] = temp[1]
			Pddotint[i] = temp[2]




		

		P = vstack((zeros((1,num-1)), P))
      
		Pdot = vstack((zeros((1,num-1)), Pdot))

		Pddot = vstack((zeros((1,num-1)), Pddot))

		
		P = hstack((Pint, P))
		Pdot = hstack((Pdotint, Pdot))
		Pddot = hstack((Pddotint, Pddot))
		# print shape(P), shape(Pdot), shape(Pddot)

		return P, Pdot, Pddot

			