import csv
import os
#The issue for why da,e is not giving me the right value is because Im not considering the fact that I saw it before
#and instead treating it as its own state and action that I have not seen before which is incorrect
#what I need to do is create a dictionary that stores the state action and qvalue similar to my first trial of this code
#along with the array that I am using. basically the dictionary will only exist in this function and be manipulated in the
#temporal difference qlearning function ONLY. If I want to change that I can simple return the dictionary of qvalues
#and make it so that the qvalues function when given a state and an action just looks up the key 
def exploration(stored_list, gamma, alpha):
  stored_list[0][2] = -1
  dict_of_qvalues = {}
  for i in range(1): #551 basically I want to make it so that if qvalues of the previous try at the first element are equal to the qvalues of this try then end it at this trial.
    for i in range(len(stored_list)-1):
      # print(stored_list[i][2])
      prev_state = stored_list[i][0]
      curr_state = stored_list[i+1][0]
      prev_action = stored_list[i][1]
      curr_action = stored_list[i+1][1]
      if(prev_state in dict_of_qvalues):
         if(prev_action in dict_of_qvalues[prev_state]):
          prev_qvalue = stored_list[i][2]
      if(curr_state in dict_of_qvalues):
        if(curr_action in dict_of_qvalues[curr_state]):
          curr_qvalue = stored_list[i+1][2]
      
      temp_dict = {}
      # if prev_action in temp_dict[] 
      temp_dict[prev_state] = prev_qvalue
      dict_of_qvalues[prev_state] = temp_dict
      temp_dict = {}
      temp_dict[curr_action] = curr_qvalue
      dict_of_qvalues[curr_state] = temp_dict
      

      stored_list[i][2], dict_of_qvalues = temporal_difference_qlearning(gamma, alpha,dict_of_qvalues,prev_state,curr_state,prev_action,curr_action,prev_qvalue,curr_qvalue)
  print(dict_of_qvalues)


def temporal_difference_qlearning(gamma, alpha,dict_of_qvalues,prev_state,curr_state,prev_action,curr_action):
  # action_qvalues = dict()
  # for action in actions:
  #   action_qvalues[action] = qvalues[state, action]


  # error_term = reward(prev_state) + gamma * max(action_qvalues) - qvalues[prev_state, prev_action]
  # qvalues[prev_state, prev_action] = qvalues[prev_state, prev_action] + alpha * (error_term)
  prev_qvalue = dict_of_qvalues[prev_state][prev_action]
  curr_qvalue = dict_of_qvalues[curr_state][curr_action]

  if(curr_qvalue == None):
     curr_qvalue = -1
  if(prev_qvalue == None):
     prev_qvalue = -1
  # print("error term is: " , -1 , "+" , gamma , "*" , "(" , prev_qvalue  , "-" ,  curr_qvalue,")")
  error_term = -1 + ((gamma * (curr_qvalue)) - prev_qvalue)
  qvalue = prev_qvalue + (alpha * error_term)

  dict_of_qvalues[prev_state][prev_action] = qvalue

  return qvalue, dict_of_qvalues
  
  # optimal_action = argmax(action_qvalues)
  # return optimal_action

class td_qlearning:
    actions = []
    states = []
    stateAndAction = []
    alpha = 0.10
    gamma = 0.95
    s = ''
    a = ''
    q = 0
    
    def __init__(self, directory):
      self.q = 0
      # self.actions
      # self.states
      index = 0
      self.stateAndAction = []
      for filename in os.listdir(directory):
        if filename.endswith('.csv'):
          filepath = os.path.join(directory, filename)
          with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
              # temp= []
              
              self.stateAndAction.append([row[0]])
              self.stateAndAction[index].append(row[1])
              self.stateAndAction[index].append(None)  
              if(row[0][0] == row[0][1]):
                self.stateAndAction[index][2] = (-10)  
              if(row[0][0] == 'B'):
                  self.stateAndAction[index][2] = (+10)  
              index = index + 1
      exploration(self.stateAndAction, self.gamma, self.alpha)     
      # for():
      #   print()
      
      

                        
                        
    def qvalue(self, state, action):
        print(self.stateAndAction)
        return self.q

def main():
    td_qlearning(directory="C:/Users/abdul/Downloads/3106A3/Examples/Example0/Trials").qvalue('DA', 'E')

main()
