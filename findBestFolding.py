## A Binary method to find the best level of folding given repeated attempts
from math import floor
from NetworkSynthesis import NetworkSynthesis
## ================================= MNIST ========================= ##
# Types of networks
# Array of the following:
# 0. number of bits
# 1. max value to use

networkTypes = [[2,1000],[8,1000]]

NetworkSynth = NetworkSynthesis()
for network in networkTypes:
    try:
        min_folding = 1
        max_folding = network[1]
        i = 0

        while not min_folding+1 == max_folding and i < 99:
                i = i+1 # Escape the while loop if error

                current_folding = floor((max_folding+min_folding)/2)
                print(f"Checking Folding value of: {current_folding}, with max and min being {max_folding},{min_folding}. Iteration number {i}")
                try:
                    NetworkSynth.networkSynthesisMNIST(current_folding,network[0])
                    # The network was able to be synthesised onto the FPGA, Now try again with a smaller folding amount
                    max_folding = current_folding
                except:
                    # Not enough folding done, network cannot fit of the chip
                    min_folding = current_folding 
        print(f'The best folding amount for {network[0]} was found to be {current_folding}')  
    except:
        pass
      

    # if synth_check: 
    #     # The folding value is still able to work
    #     max_folding = current_folding
    # else:
    #     # To not enough folding done, network cannot fit of the chip
    #     min_folding = current_folding

