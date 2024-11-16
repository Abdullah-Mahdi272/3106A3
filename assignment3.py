import csv
import os

def exploration(stored_list, gamma, alpha, dict_of_qvalues):
    # Initialize convergence tracking
    previous_qvalues = {state: actions.copy() for state, actions in dict_of_qvalues.items()}
    convergence_threshold = 1e-9  # Convergence threshold for Q-value changes
    index = 0
    #Loop until convergance
    while True:
        for i in range(len(stored_list) - 1):
            prev_state, prev_action, _ = stored_list[i]
            curr_state, curr_action, _ = stored_list[i + 1]

            # Initialize if not already present
            if prev_state not in dict_of_qvalues:
                dict_of_qvalues[prev_state] = {}
            if prev_action not in dict_of_qvalues[prev_state]:
                dict_of_qvalues[prev_state][prev_action] = stored_list[i][2] or -1

            if curr_state not in dict_of_qvalues:
                dict_of_qvalues[curr_state] = {}
            if curr_action not in dict_of_qvalues[curr_state]:
                dict_of_qvalues[curr_state][curr_action] = stored_list[i + 1][2] or -1

            # Update Q-value using temporal difference learning
            stored_list[i][2], dict_of_qvalues = temporal_difference_qlearning(
                gamma, alpha, dict_of_qvalues, prev_state, curr_state, prev_action
            )
        index = index + 1 

        # Check for convergence
        has_converged = True
        for state in dict_of_qvalues:
            for action in dict_of_qvalues[state]:
                prev_value = previous_qvalues.get(state, {}).get(action, float('inf'))
                curr_value = dict_of_qvalues[state][action]
                if abs(curr_value - prev_value) >= convergence_threshold:
                    has_converged = False
                    break
            if not has_converged:
                break

        # If all Q-values have converged, exit the loop
        if has_converged:
            break
        else:
            # Update previous Q-values for the next iteration
            previous_qvalues = {state: actions.copy() for state, actions in dict_of_qvalues.items()}

    return dict_of_qvalues


def temporal_difference_qlearning(gamma, alpha, dict_of_qvalues, prev_state, curr_state, prev_action):
    # Handle unseen actions by initializing or borrowing similar Q-values
    if prev_action not in dict_of_qvalues[prev_state]:
        dict_of_qvalues[prev_state][prev_action] = -1  # or a heuristic value if available
    
    prev_qvalue = dict_of_qvalues[prev_state][prev_action]
    max_q_next = max(dict_of_qvalues[curr_state].values(), default=0)
    error_term = -1 + gamma * max_q_next - prev_qvalue
    qvalue = prev_qvalue + alpha * error_term

    # Update Q-value
    dict_of_qvalues[prev_state][prev_action] = qvalue

    return qvalue, dict_of_qvalues




class td_qlearning:
    actions = []
    states = []
    stateAndAction = []
    trial = []
    dict_of_qvalues = {}
    alpha = 0.10
    gamma = 0.95
    s = ''
    a = ''
    q = 0
    all_dicts = []
    
    def __init__(self, directory):
      self.q = 0
      self.trial = []
      index = 0
      self.stateAndAction = []
      for filename in os.listdir(directory):
        if filename.endswith('.csv'):
          filepath = os.path.join(directory, filename)
          with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
              #Add all states in the first row to a list
              self.stateAndAction.append([row[0]])
              #For every state, store it's appropriate action inside another array
              self.stateAndAction[index].append(row[1])
              #set it's qvalue to none
              self.stateAndAction[index].append(None)  
              #set the qvalue to the reward of +10 or -10 depending on the case
              if(row[0][0] == row[0][1]):
                self.stateAndAction[index][2] = (-10)  
              if(row[0][0] == 'B'):
                  self.stateAndAction[index][2] = (+10)  
              index = index + 1
              if(row[0][0] == 'B' or row[0][0] == row[0][1] ):
                 #explore the trial we are at if we reach a terminal state
                 exploration(self.stateAndAction, self.gamma, self.alpha, self.dict_of_qvalues)
                 #store the qvalues of this trial into another array then clear the dictionary.
                 self.all_dicts.append(self.dict_of_qvalues)
                 self.stateAndAction = []
                 index = 0
                             
                        
    def qvalue(self, state, action):
        #check if a state and action is in any of the trials
        if(state in self.dict_of_qvalues):
            if(action in self.dict_of_qvalues[state]):
                #if yes the qvalue returned will be that of the states
                self.q = self.dict_of_qvalues[state][action]
        else:
            #If not then return the reward. r(s) = -10 if P_mouse == P_cat and P_mouse != P_cat
            if(state[0] == state[1] and state[0] != 'B'):
                return -10
            #r(s) = +10 if P_mouse == B
            elif(state[0] == 'B' ):
                return 10
            #else r(s) = -1
            else:
                return -1
                
        return self.q

    def policy(self, state):
        # Check if the state exists in the Q-values dictionary
        if state in self.dict_of_qvalues:
            # Find the action (key) with the maximum Q-value
            max_action = max(self.dict_of_qvalues[state], key=self.dict_of_qvalues[state].get)
            return max_action
        else:
            return "N"

#Main function for testing.

# def main():
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example0/Trials").qvalue('DA', 'E'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example0/Trials").policy('EB'))
    
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").qvalue('FB', 'D'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").qvalue('AC', 'D'))

#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").qvalue('EF', 'N'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('FC'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('FC'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('AC'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('FC'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('FB'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('AA'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('AA'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example1/Trials").policy('EF'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").qvalue('DC', 'E'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").qvalue('AE', 'B'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").qvalue('EC', 'N'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").policy('DC'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").policy('AE'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").policy('DA'))
#     print(td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example2/Trials").policy('EC'))



# main()