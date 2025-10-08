import gymnasium as gym
from gymnasium import spaces
import pyshark
import time
import socket
import numpy as np
import os
import random
import subprocess
import binascii
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
import tensorflow as tf
from tf_agents.environments import gymnasium_wrapper
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import epsilon_greedy_policy,TFPolicy
from tf_agents.replay_buffers import TFUniformReplayBuffer 
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils

from tf_agents.replay_buffers import episodic_replay_buffer

from tf_agents.networks import sequential as sequential_lib

from tf_agents.networks import sequential

from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
import threading
from datetime import datetime
# from outputUpdate import update_log_to_server 
import argparse
from fieldList import mutableFields
from config import *

# Initialize the parser
parser = argparse.ArgumentParser(description="A script that takes console arguments.")

# Add arguments
parser.add_argument('-i', '--inputpacket', type=str, required=True, help="input Packet")
parser.add_argument('-invokeID', '--invokeID', type=int, required=False, help="invokeID Field Number")
parser.add_argument('-domainID', '--domainID', type=int, required=False, help="domainID Field Number")
parser.add_argument('-itemID', '--itemID', type=int, required=False, help="=itemID Field Number")
parser.add_argument('-data', '--data', type=int, required=False, help="data Field Number")
parser.add_argument('-isInit', '--init', type=int, required=False, help=" Init Request Fuzzing enabled")

args = parser.parse_args()
# datetime object containing current date and time

env_name = "FuuzerEnv" # @param {type:"string"}

initial_collect_steps = 1 # @param {type:"integer"}
collect_steps_per_iteration =   15# @param {type:"integer"}
mutation_seq_count=15
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 1 # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}
log_interval = 4  # @param {type:"integer"}

num_eval_episodes = 1  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
replay_buffer_capacity = 15
n_step_update = 2  
num_iterations=100
num_mutations=11
response_log_interval=2

fieldPositionMap={
    "invokeID":0,
    "domainId":1,
    "itemId":2,
    "listOfData":3
}
responsesReceived=set()
text_vocab=[
        "84ff", "ffff", "8100", "21", "0300", "a0", "a1", "a2", 
        "a3", "a4", "a5", "a6", "a7", "a8", "a9", "1a", "02", "00", "ff", "81", "84", "11", 
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]

def printArrayWithIndex(arr):
    outputStr=["" for i in range(num_actions)]
    fields=mutableFields*2
    mutations=["replay","fill 0","fill 1","fill f","fill 81","empty","fill all bits","flip random bit","flip 4 bits","swap random bit","swap 4 bits"]
    fieldStr="".join([i.ljust(35) for i in fields])
    writeOutput(fieldStr+'\n')
    sum=0
    count=0
    print(len(arr))
    for i in range(len(arr)):
        outputStr[i%num_mutations]=outputStr[i%num_mutations]+(str(arr[i])+","+str(count)).ljust(35)
        sum+=arr[i]
        count+=1
    for i in range(num_mutations):
        writeOutput(mutations[i].ljust(35)+"\t"+outputStr[i]+'\n')
    writeOutput('SUM of PROB: '+str(sum/num_actions)+'\n')

def writeOutput(out_string):
    outFile=open(rootdir+'\\output\\ABB\\sritest.out','a')
    outFile.write(out_string)
    outFile.close()

def writeResponses(out_string):
    outFile=open(rootdir+'\\output\\ABB\\sritestresponses.out','a')
    outFile.write(out_string)
    outFile.close()

def writeLogs(out_string):
    outFile=open(rootdir+'\\output\\ABB\\logs.out','a')
    outFile.write(out_string)
    outFile.close()

# Calculate the edit distance of two string   
def EditDistanceRecursive(str1, str2):
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)
    return edit[len(str1)][len(str2)]


# Calculate the similarity score of two string
def SimilarityScore(str1, str2):
    ED = EditDistanceRecursive(str1, str2)
    return round((1 - (ED / max(len(str1), len(str2)))) * 100, 2)

def fillInteresting(input_string,start,end,strToFill):
    input_string=input_string[:start]+strToFill*(int((end-start)/len(strToFill)))+input_string[end:]
    return input_string

def flipAllBits(input_string,start,end):#equivalent to flipAllNibbles
    for o in range(start,end):
        binaryVal="{0:08b}".format(int(input_string[o],16))
        res=""
        for charCurIndex in range(0,len(binaryVal)):
            if binaryVal[charCurIndex]=='1':
                res+='0'
            else:
                res+='1'
        input_string=input_string[:o]+hex(int(res,2))[2:]+input_string[o+1:]
    return input_string 

def flipBit(input_string,start,end):
    flipIndex=random.randint(start,end-1)
    binaryVal="{0:08b}".format(int(input_string[flipIndex],16))
    charCurIndex=random.randint(0,len(binaryVal)-1)
    if binaryVal[charCurIndex]=='1':
        binaryVal=binaryVal[:charCurIndex]+'0'+binaryVal[charCurIndex+1:]
    else:
        binaryVal=binaryVal[:charCurIndex]+'1'+binaryVal[charCurIndex+1:]
    input_string=input_string[:flipIndex]+hex(int(binaryVal,2))[2]+input_string[flipIndex+1:]
    return input_string 

def flipNibble(input_string,start,end):
    flipIndex=random.randint(start,end-1)
    input_string=input_string[:flipIndex]+hex(int('f',16)-int(input_string[flipIndex],16))[2]+input_string[flipIndex+1:]
    return input_string 

def swapNibble(input_string,start,end):
    swapIndex1=random.randint(start,end-1)
    swapIndex2=random.randint(start,end-1)
    tempValue=str(input_string[swapIndex1])
    # len(input_string)
    input_string=input_string[:swapIndex1]+input_string[swapIndex2]+input_string[swapIndex1+1:]
    # len(input_string)
    input_string=input_string[:swapIndex2]+tempValue+input_string[swapIndex2+1:]
    # len(input_string)
    return input_string

def swapBit(input_string,start,end):
    swapCharIndex=random.randint(start, end-1)
    swapChar=input_string[swapCharIndex]
    binaryVal="{0:08b}".format(int(swapChar,16))
    swapIndex1=random.randint(0,len(binaryVal)-1)
    swapIndex2=random.randint(0,len(binaryVal)-1)
    tempValue=binaryVal[swapIndex1]
    binaryVal= binaryVal[:swapIndex1]+binaryVal[swapIndex2]+ binaryVal[swapIndex1+1:]
    binaryVal=binaryVal[:swapIndex2]+tempValue+binaryVal[swapIndex2+1:]
    input_string=input_string[:swapCharIndex]+hex(int(binaryVal,2))[2:]+input_string[swapCharIndex+1:]
    return input_string




    


class FuzzerEnv(gym.Env):
    def ping(self):
        # try:
        self.output = subprocess.getoutput('ping -n 3 {}'.format(self.ip_address))
        
            # pipe=os.popen('ping -n 15 {}'.format(self.ip_address))
            # self.output=pipe.read()
            # pipe._proc.kill()
            # pipe.close()
        # except Exception:
        #     time.sleep(300)
        #     self.ping()
        #     pipe.truncate
        # child_closed=pipe._proc.poll() is None
        # print('proc closed')

    def sendPacket(self,samp):
        words=samp
        bytearr = bytes()
        try:
            if(len(words)%2==1):
                words=words+'0'
            bytearr = bytearr + bytes.fromhex(words)
        except Exception as e:
            print('MALFORMED INPUT',e)

        # print('bytes sent: ', bytearr)
        # print(len(bytearr))
        # bytearr[3]=len(bytearr)
        try:
            (self.sck).send(bytearr)
        except:
            print('SOCKET CLOSED')
            self.setUpConnect()

        # self.messagesSent.append(bytearr)
        return 0
    
    def setUpConnect(self):
        self.sck.close()
        self.sck= socket.socket()
        self.sck.connect((self.ip_address,self.port))
        cotpReq='0300001611e00000000800c0010ac2020001c1020000'
        initReq='030000ce02f0800dc50506130100160102140200023302000134020001c1af3181aca003800101a281a4810400000001820400000001a423300f0201010604520100013004060251013010020103060528ca2202013004060251016171306f020101a06a6068a107060528ca220203a20706052901876701a30302010ca606060429018767a70302010c8a0204808b03520301ac088006303030303030be2f282d020103a028a826800300fde881010a82010a830105a416800101810305f100820c03ee1c00000408000079ef18'

        self.sendPacket(cotpReq)
        if not args.init==1:
            self.sendPacket(initReq)
    
    def fetch_fields_for_mutation(self,packet_layer):
        for fieldObj in packet_layer._all_fields:
            if(packet_layer.__getattr__(fieldObj).raw_value!=None):
                hexVal=packet_layer.__getattr__(fieldObj).binary_value
                if (len(hexVal)>0):
                    curField=packet_layer.get_field(fieldObj).all_fields[0]
                    self.non_empty_fields.append(curField)
                    # print(dir(curField))
                    # print(curField.showname_key)
                    fieldPositionMap[curField.showname_key]=self.fieldIndex
                    self.fieldIndex+=1

                    lenOf_lenField=len(hex(int(curField.size))[2:])
                    start=int(curField.pos)*2 - self.start_of_tpkt - lenOf_lenField
                    end=int(curField.pos)*2- self.start_of_tpkt
                    end2=int(curField.pos)*2+int(curField.size)*2  - self.start_of_tpkt

                    # print("Start and end of Length fields of  "+str(curField.showname_key)+" are: s="+str(start)+", e="+str(end)+"end 2="+str(end2))
                    
                    if curField.showname_key in mutableFields:
                        # For Length of Fields
                        if load_variable("action_mask") is None:
                            for mutation_id in range(self.num_mutations):
                                action_mask[mutableFields.index(curField.showname_key)*self.num_mutations+mutation_id]=1
                            # For Value of Fields
                            for mutation_id in range(self.num_mutations):
                                action_mask[len(mutableFields)*self.num_mutations+mutableFields.index(curField.showname_key)*self.num_mutations+mutation_id]=1
                        if(curField.raw_value not in text_vocab):
                            text_vocab.insert(0,curField.raw_value)

    def __init__(self):
        # SIEMENS
        self.input_string='0300002202f0800100010061153013020103a00ea00c020202babf4d05a00319012f'
        self.input_string=args.inputpacket
        #ABB READ
        # self.input_string='0300004e02f080010001006141303f020103a03aa03802015ea433800101a12ea02c302aa028a1261a0d4141314a3151303141314c44301a154c4c4e302442522472636253656375726974793031'
        #ABB GETNAMELIST
        # self.input_string='0300002402f0800100010061173015020103a010a00e020101a109a003800109a1028000'
        self.init_length=len(self.input_string)
        self.output="Lost = 0"
        # self.ip_address='127.0.0.1'
        # self.ip_address='172.18.74.71'
        self.ip_address='192.168.1.10'
        self.port = 102
        self.sck= socket.socket()
        self.num_mutations=11

        self.sck.connect((self.ip_address,self.port))
        cotpReq='0300001611e00000000800c0010ac2020001c1020000'
        initReq='030000ce02f0800dc50506130100160102140200023302000134020001c1af3181aca003800101a281a4810400000001820400000001a423300f0201010604520100013004060251013010020103060528ca2202013004060251016171306f020101a06a6068a107060528ca220203a20706052901876701a30302010ca606060429018767a70302010c8a0204808b03520301ac088006303030303030be2f282d020103a028a826800300fde881010a82010a830105a416800101810305f100820c03ee1c00000408000079ef18'

        self.sendPacket(cotpReq)
        if not args.init==1:
            self.sendPacket(initReq)

        #change response string
        self.observation_space=spaces.Text(max_length=300, charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        self.observation_space._shape=()
        
        self.malformed_action_space=spaces.Discrete(4)
        self.responsePool = []
        self.similarityScore = []
        seed_res1=str(self.sendReq()[1])
        seed_res2=str(self.sendReq()[1])
        self.responsePool.append(seed_res2)
        resp12Similarity=SimilarityScore(seed_res1.strip(), seed_res2.strip())
        if resp12Similarity>=95:
            self.similarityScore.append(resp12Similarity)
        else:
            response3=str(self.sendReq()[1])
            self.similarityScore.append(SimilarityScore(seed_res2.strip(),response3.strip()))
        responsesReceived.add(seed_res1)
        responsesReceived.add(seed_res2)

        self.action_space = spaces.Discrete(2*len(mutableFields)*self.num_mutations)
        self.start_of_tpkt=int(self.currentRequestPacket.tpkt.get_field('version').all_fields[0].pos)*2
        self.currentReward=0
        self.non_empty_fields=[]
        self.fieldIndex=0
        self.fetch_fields_for_mutation(self.currentRequestPacket.mms)
        # print(self.currentRequestPacket)
        if "acse" in dir(self.currentRequestPacket):
            self.fetch_fields_for_mutation(self.currentRequestPacket.acse)
        if "pres" in dir(self.currentRequestPacket):
            self.fetch_fields_for_mutation(self.currentRequestPacket.pres)
        
       
            
        # print(fieldPositionMap)
        # self.action_space = spaces.Discrete(len(self.non_empty_fields),start=4)
        # print("Fields in the Request",self.non_empty_fields)
        # self.action_space = spaces.Discrete(30,start=60)
        
        # sizeOfField=len(current_req_mms.get_field('itemId').all_fields[0].binary_value)
        # print('size of itemId',sizeOfField)
        # self.action_space = spaces.Discrete(4,start=81)
        # print(self.non_empty_fields)
        # print(text_vocab)
        self.it_count=0
         
    def step(self, action):
    #   print(self.currentRequestPacket)
      self.it_count+=1
      field_index=(int)(action%(len(mutableFields)*self.num_mutations)/self.num_mutations)
      field_id=fieldPositionMap[mutableFields[field_index]]
      if action<len(mutableFields)*self.num_mutations:#length field mutations
        lenOf_lenField=len(hex(int(self.non_empty_fields[field_id].size))[2:])
        if lenOf_lenField%2==1:
            lenOf_lenField+=1
        start=int(self.non_empty_fields[field_id].pos)*2 - self.start_of_tpkt - lenOf_lenField
        end=int(self.non_empty_fields[field_id].pos)*2- self.start_of_tpkt

      else:
        start=int(self.non_empty_fields[field_id].pos)*2 - self.start_of_tpkt
        end=int(self.non_empty_fields[field_id].pos)*2+int(self.non_empty_fields[field_id].size)*2  - self.start_of_tpkt

        # if action<6*self.num_mutations and action>=5*self.num_mutations:
        #     start=end
        #     end=end+2

    #   print('start of tpkt',self.start_of_tpkt)
    #   print('start of field',int(self.non_empty_fields[field_id].pos)*2)
    #   print('size of field',int(self.non_empty_fields[field_id].size)*2)
    #   print('start',start)
    #   print('end',end)
      if action%self.num_mutations==1:
          self.input_string=fillInteresting(self.input_string,start,end,'0')
          
      elif action%self.num_mutations==2:
          self.input_string=fillInteresting(self.input_string,start,end,'1')
      elif action%self.num_mutations==3:
          self.input_string=fillInteresting(self.input_string,start,end,'f')
      elif action%self.num_mutations==4:
          self.input_string=fillInteresting(self.input_string,start,end,'81')
      elif action%self.num_mutations==5: #empty
          self.input_string=fillInteresting(self.input_string,start,end,'0')
        #   self.input_string=self.input_string[:start]+self.input_string[end:]
      elif action%self.num_mutations==6: 
          self.input_string=flipAllBits(self.input_string,start,end)
      elif action%self.num_mutations==7:
          self.input_string=flipBit(self.input_string,start,end)
      elif action%self.num_mutations==8:
          self.input_string=flipNibble(self.input_string,start,end)
      elif action%self.num_mutations==9:
          self.input_string=swapBit(self.input_string,start,end)
      elif action%self.num_mutations==10:
          self.input_string=swapNibble(self.input_string,start,end)

      self.input_string=self.input_string[:self.init_length]
      proc=threading.Thread(target=self.ping)
      proc.start()
      observation=self._get_obs()
      
      response1 = str(self.sendReq()[1])
      response2=str(self.sendReq()[1])
      proc.join()
      writeOutput(str(self.sendReq())+'\n')
      info = self._get_info() 
        # An episode is done iff the agent has reached the target
      terminated = not info
      reward=0
    #   print(self.input_string)
    #   print(response1)
      unexploredFlag=True

      if terminated:
        writeOutput("Crash Encountered!!! on "+str(datetime.now())+ "\nInput String:"+self.input_string+"\n")
        reward=20
        time.sleep(200)
      else:
        for j in range(0, len(self.responsePool)):
            target = self.responsePool[j]
            score = self.similarityScore[j]
            c = SimilarityScore(target.strip(), response1.strip())
            if c >= score:
                # if c<50 and resSimScores>=95:
                #     continue
                unexploredFlag = False
                reward=0
                break
        if unexploredFlag:
            if response1 not in responsesReceived:
                writeResponses('New Response Received at Time: '+str(datetime.now())+'\n')
                writeResponses(response1+'\n')
            responsesReceived.add(response1)
            self.responsePool.append(response1)
            self.similarityScore.append(SimilarityScore(response1.strip(), response2.strip()))
            reward=1
        # if "confirmed_errorpdu_element" in response1:
        #     reward=7.5
        if "Malformed" in response1:
            terminated=True
            writeOutput("Malformed Response or Information Leak Encountered!!! on "+str(datetime.now())+ "\nInput String:"+self.input_string+"\n")
            reward=20
        if "timeout" in response1:
            reward=10

            # WRITE CODE FOR getting obs and calculating reward, getting info and identifying crash to terminate
        
      writeOutput(f'Reward={reward}, action chosen:{action} , field chosen: {self.non_empty_fields[field_id]}\n')
      return observation, reward, terminated, self.it_count==15, info
    
    def _get_obs(self): 
        return self.input_string
        # return tf.keras.random.uniform((64,1), minval=0.0, maxval=1.0, dtype=None, seed=69)
    
    def sendReq(self): 
        # cap = pyshark.LiveCapture(interface="Ethernet 2",custom_parameters={"-C": "tshark-mms"},tshark_path='/usr/local/bin/tshark')
        # cap = pyshark.LiveCapture(interface="Ethernet",bpf_filter="tcp port 102")
        try:
            cap = pyshark.LiveCapture(interface="Ethernet")
        except Exception as exceptionDetail:
            self.setUpConnect()
            return self.input_string,str("#Capture Not setup" + str(exceptionDetail))
        # cap = pyshark.LiveCapture(bpf_filter="tcp port 102")
        try:
            count=0
            timeout=15
            for packet in cap.sniff_continuously():
                if count==0:
                    retVal=self.sendPacket(self.input_string)
                    start = time.time()
                    if retVal==1:
                        print("Error while sending request")
                        cap.close()
                        return self.input_string,'#error'
                    self.currentRequestPacket=None
                    count=1
                if("tpkt" in dir(packet) and str(packet.ip.dst) in self.ip_address):
                    # print(packet)
                    self.currentRequestPacket=packet  
                if("mms" in dir(packet)):    
                    # if "confirmedservicerequest" or "initiate_requestpdu_element" in dir(packet.mms):
                    #     self.currentRequestPacket=packet            
                    if "rejectpdu_element" in dir(packet.mms):
                        cap.close()
                        return self.input_string,str(packet.mms)
                    if args.init==1 and "initiate_responsepdu_element" in dir(packet.mms):
                        # self.closeCapture()
                        cap.close()
                        self.setUpConnect()
                        return self.input_string,str(packet.mms)
                    if "confirmedserviceresponse" in dir(packet.mms):
                        cap.close()
                        return self.input_string,str(packet.mms)
                    if "confirmed_errorpdu_element" in dir(packet.mms):
                        cap.close()
                        return self.input_string,str(packet.mms)
                    
                elif("ip" in dir(packet) and str(packet.ip.src) in self.ip_address):
                    if("mms" in dir(packet)):
                        cap.close()
                        return self.input_string,str(packet.mms)
                    elif("acse" in dir(packet)):
                        cap.close()
                        self.setUpConnect()
                        return self.input_string,str(packet.acse)
                    elif("pres" in dir(packet)):
                        cap.close()
                        self.setUpConnect()
                        return self.input_string,str(packet.pres)
                    elif("ses" in dir(packet)):
                        cap.close()
                        self.setUpConnect()
                        return self.input_string,str(packet.ses)
            
                if timeout and time.time() - start > timeout:
                    cap.close()
                    self.setUpConnect()
                    # self.sck.close()
                    # self.__init__()
                    return self.input_string,'#timeout'
            
            cap.close()
            self.setUpConnect()
            return self.input_string,'#timeout'
        except ConnectionResetError:
            self.setUpConnect()
            cap.close()
            return self.input_string,"#socket_closed - remote host forcibly closed existing connection - Connection Reset"
        except ConnectionAbortedError :
            self.setUpConnect()
            cap.close()
            return self.input_string,"#socket_closed - connection Aborted"
        except Exception as exceptionDetail:
            print(exceptionDetail)
            self.setUpConnect()
            cap.close()
            return self.input_string,"#Capture error - Pass"
    

    def _get_info(self):
        #write code for ping should run just after get obs, or track fro a file if it has carshed in the interval
        # print(self.output)
        if "Lost = 0" in self.output:
            return True
        return False

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #send new seed request
        self.sck.close()
        self.__init__()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    def close(self):
        self.sck.close()


# rootdir="C:\\Users\\Kanmani A\\Documents\\PowerGrid\\Snipuzz-py\\rlImpl\\"
rootdir = "C:\\Users\\user\\Desktop\\sridevi\\Snipuzz-py-mutation\\rlImpl\\"
# tempdir="C:\\Users\\Kanmani A\\Documents\\PowerGrid\\Snipuzz-py\\rlImpl\\temp"
tempdir="C:\\Users\\user\\Desktop\\sridevi\\Snipuzz-py-mutation\\rlImpl\\temp"
# testdir="C:\\Users\\Kanmani A\\Documents\\PowerGrid\\Snipuzz-py\\rlImpl\\test"
testdir="C:\\Users\\user\\Desktop\\sridevi\\Snipuzz-py-mutation\\rlImpl\\test"
tf.compat.v1.enable_v2_behavior()
train_dir = os.path.join(rootdir, 'train')
eval_dir = os.path.join(testdir, 'REC670InitiateAuth1')


train_summary_writer = tf.compat.v2.summary.create_file_writer(
    train_dir, flush_millis=10 * 1000
)
train_summary_writer.set_as_default()

eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    eval_dir, flush_millis=10 * 1000
)

def compute_avg_return(environment, policy, num_episodes=2):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    it_count=0
    # while not time_step.is_last():
    while it_count<mutation_seq_count and not time_step.is_last():
      it_count+=1
      prev_time_step = environment.current_time_step()
      action_step = policy.action(time_step)
    #   writeQvalues(prev_time_step,action_step)
      time_step = environment.step(action_step.action)

    #   traj = trajectory.from_transition(prev_time_step, action_step, time_step)
      episode_return += time_step.reward
      q_values, _ = agent._q_network(prev_time_step.observation)
      

    total_return += episode_return

    if it_count!=mutation_seq_count:
        writeOutput("Masking Action No. "+str(action_step.action.numpy()[0])+"\n")
        action_mask[action_step.action.numpy()[0]]=0
        save_variable("action_mask",action_mask)

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

eval_metrics = [
    tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes)
]

if load_variable("action_mask") is None:
    action_mask = [0 for i in range(2*len(mutableFields)*11)]
else:
    action_mask=load_variable("action_mask")
def observation_and_action_constraint_splitter(observation):
    actionStep=tf.convert_to_tensor(action_mask, dtype=tf.int64)
    return observation, actionStep
    
writeOutput('=================================================================================================\n')
writeOutput('Start Time: '+str(datetime.now())+'\n')

train_env_py=FuzzerEnv()
eval_env_py=FuzzerEnv()
train_env = tf_py_environment.TFPyEnvironment(gymnasium_wrapper.GymnasiumWrapper(train_env_py))
eval_env = tf_py_environment.TFPyEnvironment(gymnasium_wrapper.GymnasiumWrapper(eval_env_py))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
fc_layer_params = ( 256, 128)
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
observation_tensor_spec=tensor_spec.from_spec(train_env.observation_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

#update_log_to_server("C:\\Users\\Kanmani A\\Documents\\PowerGrid\\Snipuzz-py\\rlImpl\\output\\Siemens\\testRes.out","logRLStepsSimpleModel.out")
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

# Define the positional embedding layer
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = self.pos_embedding(positions)
        return x + positions

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
fc_layer_params = (256, 512, 512, 256)  # Example dense layer configuration
dense_layers = []
for num_units in fc_layer_params:
    dense_layers.append(tf.keras.layers.Dense(num_units, activation='relu'))
    dense_layers.append(tf.keras.layers.Dropout(0.2))
print('num_actions',num_actions)
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    # kernel_initializer=tf.keras.initializers.Constant(1),
    # bias_initializer=tf.keras.initializers.Constant(1)
    kernel_initializer=tf.keras.initializers.Constant(0.0),
    bias_initializer=tf.keras.initializers.Constant(0.0)
    )

max_tokens = 100  # Maximum vocab size.
max_len = 150  # Maximum sequence length
embed_dim = 64  # Embedding dimension

input_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len,
    vocabulary=text_vocab)

embedding_layer = tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=embed_dim)

# Combine the embedding with positional encoding
pos_embedding_layer = PositionalEmbedding(max_len=max_len, embed_dim=embed_dim)
pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
# Construct the final QNetwork model
q_net = sequential.Sequential([
    input_layer,
    embedding_layer,
    pos_embedding_layer,
    pooling_layer
] + dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0,dtype=tf.int64)

global_step = tf.compat.v1.train.get_or_create_global_step()

epsilon_fn = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.8,  # Starting epsilon
    decay_steps=4,          # Decay step size
    decay_rate=0.8,            # Decay rate
    staircase=False
)

# print(train_env.time_step_spec())
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
    train_step_counter=train_step_counter,
    epsilon_greedy=epsilon_fn(global_step),
    debug_summaries=True,
    summarize_grads_and_vars=True)

agent.initialize()


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec(),
                                                observation_and_action_constraint_splitter= observation_and_action_constraint_splitter
                                                )
# tf_policy=TFPolicy(train_env.time_step_spec(),train_env.action_spec())
# g_epsilon_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(tf_policy, epsilon=0.4)


print('before reset',eval_env.time_step_spec())
timestep=eval_env.reset()
# print('after reset',timestep)
# action_step = agent.collect_policy.action(timestep)
# val=action_step.action.numpy()
# print(action_step.action.numpy())
# print('after action',random_policy.time_step_spec)




checkpoint_dir = os.path.join(tempdir, 'REC670InitiateAuth1')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.collect_policy,
    global_step=global_step
)
train_checkpointer.initialize_or_restore()
q_values=agent._q_network.weights[-1].numpy()
printArrayWithIndex(q_values)

def writeQvalues(time_step,action_step):
    q_values, _ = agent._q_network(time_step.observation)

    writeOutput("Q-values:")
    # printArrayWithIndex(q_values.numpy()[0])
    nonzero_action_value={}
    q_values=q_values.numpy()[0]
    for index in range(len(q_values)):
        if(q_values[index]!=0):
            nonzero_action_value[index]=q_values[index]
    
    writeOutput(str(nonzero_action_value)+"\n")
            

    writeOutput("Selected Action:"+str(action_step.action.numpy())+"\n")
    writeOutput("Q-value of selected action on current state"+str(q_values[action_step.action.numpy()])+"\n")
# q_values_measure,_=agent._q_network
# printArrayWithIndex(q_values_measure)

# writeQvalues(timestep,action_step)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
#   if next_time_step.is_last() and policy!=agent.collect_policy:
#         action_mask[action_step.action.numpy()[0]]=0
#         save_variable("action_mask",action_mask)
  # Add trajectory to the replay buffer
#   experience = tf.nest.map_structure(
#         lambda x: x[tf.newaxis, ...],
#         [traj,traj],
#     )
  return traj

# writeOutput('Collecting Inital Time Steps...\n')
def collect_and_return_experience(environment,policy):
    prev_traj=None
    batched_trajectory=[]
    for _ in range(mutation_seq_count*4):
        cur_traj=collect_step(train_env, agent.collect_policy) # Get batch size dynamically
        if prev_traj is None:
            prev_traj=cur_traj
            continue
        traj_buf=[prev_traj,cur_traj]
        prev_traj=cur_traj
        cur_trans = tf.nest.map_structure(lambda *x:  tf.concat([x[0], x[1]], axis=0), *traj_buf)
        batched_trajectory.append(cur_trans)
        if _ %15 ==0:
            train_env.reset()

        # Stack each field (step_type, action, observation, etc.) correctly
    # batched_trajectory = tf.nest.map_structure(
    #     lambda x: tf.stack([x[i:i + 2] for i in range(3)], axis=0),
    #     traj_buf
    # )

    batched_trajectory = tf.nest.map_structure(lambda *x: tf.stack(x, axis=0), *batched_trajectory)
    # writeLogs(str(batched_trajectory))
    return batched_trajectory
    

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where





# (Optional) Optimize by wrapping some of the code in a graph using TF function.
# agent.train = common.function(agent.train)

# eps = 1.0
# decay = 0.8, step = 1
# lr = 1e-3
# train = epsilon greedy
# evaluate = agent policy

agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
# avg_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
# returns = [avg_return]
# writeOutput('before Training: '+str(returns)+'\n')
cumulative_reward=0
step=0

for step_num in range(1,num_iterations+1):

  # Collect a few steps using collect_policy and save to the replay buffer.
    #   train_env.reset()
  train_env.reset()
#   for _ in range(collect_steps_per_iteration):
#     collect_step(train_env, random_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  if step%4==0:
    train_loss=agent.train(collect_and_return_experience(train_env,agent.collect_policy))

#   step = agent.train_step_counter.numpy()
  step=step+1
#   q_values=agent._q_network.weights[-1].numpy()
#   printArrayWithIndex(q_values)

#   writeOutput('step = {0}: loss = {1}'.format(step, train_loss.loss)+'\n')
  global_step = tf.compat.v1.train.get_or_create_global_step()
  train_checkpointer.save(global_step)

  traj_buf=[]
  rewards = compute_avg_return(eval_env, agent.collect_policy, num_eval_episodes)
  
#   q_values=agent._q_network.weights[-1].numpy()
#   printArrayWithIndex(q_values)
  cumulative_reward+=rewards
  total_reward_avg=cumulative_reward/step
  writeOutput('step = {0}: Reward = {1:.2f} , Average Reward= {2:.2f}'.format(step, rewards, total_reward_avg)+'\n')
  
  


#   if step % log_interval == 0:
#     update_log_to_server()

  
#   avg_return = compute_avg_return(eval_env, agent.collect_policy, num_eval_episodes)
#   if step % eval_interval == 0:
#     writeOutput('step = {0}: Average Return = {1:.2f}'.format(step, avg_return)+'\n')

#   if step % eval_interval == 0:
#   step_var = tf.cast(step, tf.int64)
#   results = metric_utils.eager_compute(
#         eval_metrics,
#         eval_env,
#         agent.collect_policy,
#         num_episodes=num_eval_episodes,
#         train_step=step_var,
#         summary_writer=eval_summary_writer,
#         summary_prefix='Metrics',
#     )
#     # if eval_metrics_callback is not None:
#     #   eval_metrics_callback(results, global_step.numpy())
#   metric_utils.log_metrics(eval_metrics)
  if step is not None and eval_summary_writer:
    with eval_summary_writer.as_default():
        tf.compat.v2.summary.scalar(name="Reward", data=rewards, step=step)
        tf.compat.v2.summary.scalar(name="Average Reward", data=total_reward_avg, step=step)
        tf.compat.v2.summary.scalar(name="Cumulative Reward", data=cumulative_reward, step=step)
  
#   returns.append(avg_return)

writeOutput('End Time: '+str(datetime.now())+'\n')